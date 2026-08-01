"""Add cumulative CPU timing instrumentation to an installed vLLM.

Anchor verification
-------------------
The anchors in this module were checked against the upstream ``v0.22.1`` tag
(commit ``0decac0d96c42b49572498019f0a0e3600f50398``):

* ``vllm/v1/core/sched/scheduler.py``: ``Scheduler.schedule`` near line 328
  and ``Scheduler.update_from_output`` near line 1282.
* ``vllm/v1/engine/core.py``: ``EngineCore.step`` near line 427 and
  ``EngineCore.step_with_batch_queue`` near line 468.
* ``vllm/v1/worker/gpu_model_runner.py``: ``_prepare_inputs`` near line 1858
  and ``execute_model`` near line 3950.
* ``vllm/v1/engine/output_processor.py``:
  ``OutputProcessor.process_outputs`` near line 571.
* ``vllm/v1/engine/detokenizer.py``: the class-level ``update`` methods on
  ``IncrementalDetokenizer`` near line 41 and ``BaseIncrementalDetokenizer``
  near line 95.

Only ``Scheduler.schedule`` and ``EngineCore.step`` are required anchors.
Other methods are version-sensitive and are skipped with a warning when they
are absent. The patch is idempotent and ``--check`` performs no writes.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

_TIMING_SOURCE = Path(__file__).parent / "schedulers/instrumentation/timing.py"
_TIMING_DESTINATION = Path("vllm/v1/timing.py")
_TIMING_IMPORT = "from vllm.v1.timing import profile_scheduler_function"
_DECORATOR = "profile_scheduler_function"


@dataclass(frozen=True)
class MethodTarget:
    """A method anchor to find and decorate."""

    label: str
    names: tuple[str, ...]
    required: bool = False
    all_matches: bool = False


@dataclass(frozen=True)
class FileTarget:
    relative_path: Path
    methods: tuple[MethodTarget, ...]


@dataclass(frozen=True)
class AnchorReport:
    relative_path: Path
    label: str
    matched_name: str | None
    match_count: int
    decorated_count: int
    required: bool


_TARGETS = (
    FileTarget(
        Path("vllm/v1/core/sched/scheduler.py"),
        (
            MethodTarget("Scheduler.schedule", ("schedule",), required=True),
            MethodTarget("Scheduler.update_from_output", ("update_from_output",)),
        ),
    ),
    FileTarget(
        Path("vllm/v1/engine/core.py"),
        (
            MethodTarget("EngineCore.step", ("step",), required=True),
            MethodTarget("EngineCore.step_with_batch_queue", ("step_with_batch_queue",)),
        ),
    ),
    FileTarget(
        Path("vllm/v1/worker/gpu_model_runner.py"),
        (
            MethodTarget("GPUModelRunner.execute_model", ("execute_model",)),
            MethodTarget(
                "GPUModelRunner input preparation",
                (
                    "_prepare_inputs",
                    "_prepare_model_inputs",
                    "_prepare_input_tensors",
                    "prepare_inputs",
                ),
            ),
        ),
    ),
    FileTarget(
        Path("vllm/v1/engine/output_processor.py"),
        (MethodTarget("OutputProcessor.process_outputs", ("process_outputs",)),),
    ),
    FileTarget(
        Path("vllm/v1/engine/detokenizer.py"),
        (
            MethodTarget(
                "detokenizer update methods",
                ("update",),
                all_matches=True,
            ),
        ),
    ),
)


def _method_pattern(name: str) -> re.Pattern[str]:
    # vLLM's target methods are direct class members. Restricting indentation
    # to four spaces avoids accidentally decorating a nested helper.
    return re.compile(
        rf"^(?P<indent>    )(?P<async>async\s+)?def\s+{re.escape(name)}\s*\(",
        re.MULTILINE,
    )


def _matching_method(
    text: str,
    target: MethodTarget,
) -> tuple[str | None, list[re.Match[str]]]:
    for name in target.names:
        matches = list(_method_pattern(name).finditer(text))
        if matches:
            return name, matches
    return None, []


def _is_decorated(text: str, match: re.Match[str]) -> bool:
    line_start = match.start()
    preceding_line_end = line_start - 1
    if preceding_line_end < 0:
        return False
    preceding_line_start = text.rfind("\n", 0, preceding_line_end) + 1
    return text[preceding_line_start:preceding_line_end].strip() == f"@{_DECORATOR}"


def inspect_source_tree(site_packages: Path) -> list[AnchorReport]:
    """Inspect target anchors without modifying the source tree."""

    reports: list[AnchorReport] = []
    for file_target in _TARGETS:
        path = site_packages / file_target.relative_path
        if not path.is_file():
            reports.extend(
                AnchorReport(
                    file_target.relative_path,
                    method.label,
                    None,
                    0,
                    0,
                    method.required,
                )
                for method in file_target.methods
            )
            continue

        text = path.read_text()
        for method in file_target.methods:
            name, matches = _matching_method(text, method)
            reports.append(
                AnchorReport(
                    file_target.relative_path,
                    method.label,
                    name,
                    len(matches),
                    sum(_is_decorated(text, match) for match in matches),
                    method.required,
                )
            )
    return reports


def _validate_reports(reports: list[AnchorReport]) -> None:
    missing = [report for report in reports if report.required and report.match_count == 0]
    if missing:
        details = ", ".join(f"{report.label} in {report.relative_path}" for report in missing)
        raise RuntimeError(f"Required vLLM timing anchors are missing: {details}")


def _print_reports(reports: list[AnchorReport]) -> None:
    for report in reports:
        status = "MATCH" if report.match_count else ("MISSING" if report.required else "WARNING/SKIP")
        name = f" ({report.matched_name})" if report.matched_name else ""
        decorated = f", decorated={report.decorated_count}/{report.match_count}" if report.match_count else ""
        print(
            f"{status}: {report.relative_path}: {report.label}{name}; matches={report.match_count}{decorated}",
            flush=True,
        )


def _insert_timing_import(text: str, path: Path) -> str:
    if _TIMING_IMPORT in text:
        return text
    import_match = re.search(r"^(?:import vllm\b|from vllm\b)", text, re.MULTILINE)
    if import_match is None:
        raise RuntimeError(f"Could not locate a vLLM import anchor in {path}")
    return text[: import_match.start()] + _TIMING_IMPORT + "\n" + text[import_match.start() :]


def _decorate_matches(
    text: str,
    matches: list[re.Match[str]],
    *,
    all_matches: bool,
) -> str:
    selected = matches if all_matches else matches[:1]
    for match in reversed(selected):
        if _is_decorated(text, match):
            continue
        indent = match.group("indent")
        text = text[: match.start()] + f"{indent}@{_DECORATOR}\n" + text[match.start() :]
    return text


def patch_source_tree(site_packages: Path) -> list[AnchorReport]:
    """Patch an unpacked vLLM installation rooted at site-packages."""

    reports = inspect_source_tree(site_packages)
    _validate_reports(reports)

    timing_destination = site_packages / _TIMING_DESTINATION
    timing_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_TIMING_SOURCE, timing_destination)

    for file_target in _TARGETS:
        path = site_packages / file_target.relative_path
        if not path.is_file():
            continue
        original = path.read_text()
        text = original
        matched_any = False
        for method in file_target.methods:
            _, matches = _matching_method(text, method)
            if not matches:
                continue
            matched_any = True
            text = _decorate_matches(text, matches, all_matches=method.all_matches)
        if matched_any:
            text = _insert_timing_import(text, path)
        if text != original:
            path.write_text(text)

    return inspect_source_tree(site_packages)


def _site_packages(venv_python: Path) -> Path:
    if not venv_python.is_file():
        raise FileNotFoundError(f"Virtual-environment Python not found: {venv_python}")
    result = subprocess.run(
        [
            str(venv_python),
            "-c",
            "import site; print(site.getsitepackages()[0])",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    path = Path(result.stdout.strip())
    if not (path / "vllm").is_dir():
        raise FileNotFoundError(f"vLLM is not installed under {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venv-python",
        type=Path,
        required=True,
        help="Python executable in the existing vLLM virtual environment",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report anchor matches without modifying the installation",
    )
    args = parser.parse_args(argv)

    # Do not resolve the executable symlink: venv Python launchers commonly
    # point at the system interpreter, and resolving it drops the venv context.
    site_packages = _site_packages(args.venv_python.expanduser().absolute())
    reports = inspect_source_tree(site_packages)
    _print_reports(reports)
    _validate_reports(reports)
    if args.check:
        print("Check complete; no files were modified.", flush=True)
        return 0

    reports = patch_source_tree(site_packages)
    print(f"Installed timing module at {site_packages / _TIMING_DESTINATION}", flush=True)
    _print_reports(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
