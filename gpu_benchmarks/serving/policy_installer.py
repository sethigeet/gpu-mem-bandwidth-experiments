"""Install policy-enabled vLLM using schedulers vendored in this project.

This installer is GPU-host-only. It patches the official pinned vLLM wheel in
an isolated virtual environment and does not require the original research
repository.
"""

import argparse
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from gpu_benchmarks.serving.timing_installer import patch_source_tree

VENDORED_SCHEDULERS = Path(__file__).with_name("schedulers")
POLICIES = (
    "fcfs",
    "priority",
    "radix_cost",
    "chunked_hash_tree_bandit",
    "chunked_hash_tree_python",
    "chunked_hash_tree_cpp",
)
VLLM_WHEEL_VERSION = "0.14.0"


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _expose_custom_policy_choices(vllm_source: Path) -> None:
    config_path = vllm_source / "vllm/config/scheduler.py"
    text = config_path.read_text()
    values = ", ".join(f'"{policy}"' for policy in POLICIES)
    new = f"SchedulerPolicy = Literal[{values}]"
    if new in text:
        return
    pattern = r"SchedulerPolicy = Literal\[[^\n]+\]"
    if re.search(pattern, text) is None:
        raise RuntimeError(f"Could not locate SchedulerPolicy declaration in {config_path}")
    config_path.write_text(re.sub(pattern, new, text, count=1))


def _configure_request_queues(vllm_source: Path) -> None:
    request_queue_path = vllm_source / "vllm/v1/core/sched/request_queue.py"
    text = request_queue_path.read_text()

    enum_anchor = '    PRIORITY = "priority"'
    enum_replacement = (
        f"{enum_anchor}\n"
        '    RADIX_COST = "radix_cost"\n'
        '    CHUNKED_HASH_TREE_BANDIT = "chunked_hash_tree_bandit"\n'
        '    CHUNKED_HASH_TREE_PYTHON = "chunked_hash_tree_python"\n'
        '    CHUNKED_HASH_TREE_CPP = "chunked_hash_tree_cpp"'
    )
    if "RADIX_COST" not in text:
        if enum_anchor not in text:
            raise RuntimeError(f"Could not locate policy enum in {request_queue_path}")
        text = text.replace(enum_anchor, enum_replacement, 1)

    queue_defaults_marker = "    def should_add_more_to_batch(self, **kwargs) -> bool:"
    if queue_defaults_marker not in text:
        bool_anchor = "    @abstractmethod\n    def __bool__(self) -> bool:"
        queue_defaults = (
            "    def free_request(self, request: Request) -> None:\n"
            "        del request\n\n"
            "    def should_add_more_to_batch(self, **kwargs) -> bool:\n"
            "        del kwargs\n"
            "        return True\n\n"
            f"{bool_anchor}"
        )
        if bool_anchor not in text:
            raise RuntimeError(f"Could not locate RequestQueue methods in {request_queue_path}")
        text = text.replace(bool_anchor, queue_defaults, 1)

    factory_anchor = (
        "    elif policy == SchedulingPolicy.FCFS:\n"
        "        return FCFSRequestQueue()\n"
        "    else:\n"
        '        raise ValueError(f"Unknown scheduling policy: {policy}")'
    )
    if "ChunkedHashTreeBanditRequestQueue()" not in text:
        factory_replacement = (
            "    elif policy == SchedulingPolicy.FCFS:\n"
            "        return FCFSRequestQueue()\n"
            "    else:\n"
            "        from vllm.v1.core.sched.policy_request_queues import (\n"
            "            ChunkedHashTreeBanditRequestQueue,\n"
            "            CppChunkedHashTreeRequestQueue,\n"
            "            PythonChunkedHashTreeRequestQueue,\n"
            "            RadixCostRequestQueue,\n"
            "        )\n\n"
            "        custom_queues = {\n"
            "            SchedulingPolicy.RADIX_COST: RadixCostRequestQueue,\n"
            "            SchedulingPolicy.CHUNKED_HASH_TREE_BANDIT: "
            "ChunkedHashTreeBanditRequestQueue,\n"
            "            SchedulingPolicy.CHUNKED_HASH_TREE_PYTHON: "
            "PythonChunkedHashTreeRequestQueue,\n"
            "            SchedulingPolicy.CHUNKED_HASH_TREE_CPP: "
            "CppChunkedHashTreeRequestQueue,\n"
            "        }\n"
            "        queue_type = custom_queues.get(policy)\n"
            "        if queue_type is None:\n"
            '            raise ValueError(f"Unknown scheduling policy: {policy}")\n'
            "        return queue_type()"
        )
        if factory_anchor not in text:
            raise RuntimeError(f"Could not locate queue factory in {request_queue_path}")
        text = text.replace(factory_anchor, factory_replacement, 1)

    request_queue_path.write_text(text)


def _configure_scheduler(vllm_source: Path) -> None:
    scheduler_path = vllm_source / "vllm/v1/core/sched/scheduler.py"
    text = scheduler_path.read_text()

    timing_import = "from vllm.v1.core.sched.scheduler_timing import profile_scheduler_function\n"
    if timing_import not in text:
        import_anchor = "from vllm.v1.core.sched.scheduler import"
        first_scheduler_import = text.find(import_anchor)
        if first_scheduler_import == -1:
            import_anchor = "from vllm.v1.core.sched."
            first_scheduler_import = text.find(import_anchor)
        if first_scheduler_import == -1:
            raise RuntimeError(f"Could not locate scheduler imports in {scheduler_path}")
        text = text[:first_scheduler_import] + timing_import + text[first_scheduler_import:]

    timing_marker = "        self._policy_last_schedule_time: float | None = None"
    if timing_marker not in text:
        schedule_anchor = "    def schedule(self) -> SchedulerOutput:\n"
        timing_fields = (
            f"{timing_marker}\n"
            "        self._policy_batch_id = 0\n\n"
            f"{schedule_anchor}"
            "        current_schedule_time = time.time()\n"
            "        last_batch_time = (\n"
            "            None\n"
            "            if self._policy_last_schedule_time is None\n"
            "            else current_schedule_time - self._policy_last_schedule_time\n"
            "        )\n"
            "        self._policy_last_schedule_time = current_schedule_time\n"
            "        self._policy_batch_id += 1\n"
        )
        if schedule_anchor not in text:
            raise RuntimeError(f"Could not locate schedule method in {scheduler_path}")
        text = text.replace(schedule_anchor, timing_fields, 1)

    schedule_anchor = "    def schedule(self) -> SchedulerOutput:\n"
    schedule_decorator = "    @profile_scheduler_function\n"
    if schedule_decorator + schedule_anchor not in text:
        if schedule_anchor not in text:
            raise RuntimeError(f"Could not locate schedule method in {scheduler_path}")
        text = text.replace(
            schedule_anchor,
            schedule_decorator + schedule_anchor,
            1,
        )

    loop_anchor = "            while self.waiting and token_budget > 0:\n"
    gate_marker = "self.waiting.should_add_more_to_batch("
    if gate_marker not in text:
        gated_loop = (
            f"{loop_anchor}"
            "                if not self.waiting.should_add_more_to_batch(\n"
            "                    current_batch_size=len(self.running),\n"
            "                    last_batch_time=last_batch_time,\n"
            "                    current_batch_id=self._policy_batch_id,\n"
            "                ):\n"
            "                    break\n"
        )
        if loop_anchor not in text:
            raise RuntimeError(f"Could not locate waiting loop in {scheduler_path}")
        text = text.replace(loop_anchor, gated_loop, 1)

    free_anchor = "        assert request.is_finished()\n"
    free_marker = "        self.waiting.free_request(request)\n"
    if free_marker not in text:
        if free_anchor not in text:
            raise RuntimeError(f"Could not locate request cleanup in {scheduler_path}")
        text = text.replace(free_anchor, free_anchor + free_marker, 1)

    scheduler_path.write_text(text)


def _install(venv_dir: Path, python: str) -> Path:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to install the policy-enabled vLLM fork")

    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    if venv_dir.exists():
        if not (venv_dir / "pyvenv.cfg").is_file():
            raise ValueError(f"Refusing to replace non-virtual-environment directory {venv_dir}")
        shutil.rmtree(venv_dir)
    _run([uv, "venv", "--python", python, str(venv_dir)])
    venv_python = venv_dir / "bin/python"
    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(venv_python),
            "--link-mode",
            "copy",
            f"vllm=={VLLM_WHEEL_VERSION}",
            "pybind11",
            "xxhash",
        ]
    )

    site_packages = Path(
        subprocess.run(
            [
                str(venv_python),
                "-c",
                "import site; print(site.getsitepackages()[0])",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    target_sched = site_packages / "vllm/v1/core/sched"
    copied_files = {
        "chunked_hash_tree/python.py": "chunked_hash_tree_python.py",
        "chunked_hash_tree/contextual_bandit.py": "contextual_bandit.py",
        "instrumentation/timing.py": "scheduler_timing.py",
        "policy_request_queues.py": "policy_request_queues.py",
        "radix_cost/python.py": "radix_cost.py",
    }
    for source_name, target_name in copied_files.items():
        shutil.copy2(VENDORED_SCHEDULERS / source_name, target_sched / target_name)
    for filename in (
        "CMakeLists.txt",
        "bindings.cpp",
        "chunked_hash_tree.cpp",
        "chunked_hash_tree.hpp",
        "chunked_hash_tree_rl.cpp",
        "chunked_hash_tree_rl.hpp",
    ):
        shutil.copy2(VENDORED_SCHEDULERS / filename, target_sched / filename)
    _expose_custom_policy_choices(site_packages)
    _configure_request_queues(site_packages)
    _configure_scheduler(site_packages)
    patch_source_tree(site_packages)

    build_dir = target_sched / "build"
    pybind_dir = subprocess.run(
        [str(venv_python), "-m", "pybind11", "--cmakedir"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _run(
        [
            "cmake",
            "-S",
            str(target_sched),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-Dpybind11_DIR={pybind_dir}",
        ]
    )
    _run(["cmake", "--build", str(build_dir), "--parallel"])
    return venv_dir / "bin/vllm"


def _verify(executable: Path) -> None:
    result = subprocess.run(
        [str(executable), "serve", "--help=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    missing = [policy for policy in POLICIES if policy not in result.stdout]
    if missing:
        raise RuntimeError(f"Policy-enabled vLLM is missing: {', '.join(missing)}")
    _run([str(executable), "--version"])


def main(argv: list[str] | None = None) -> int:
    cache_root = Path(os.environ.get("VLLM_POLICY_CACHE", "~/.cache/gpu-memory-benchmarks/vllm")).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv-dir", type=Path, default=cache_root / "policy_venv")
    parser.add_argument("--python", default="3.12")
    parser.add_argument("--wait-for-tmux-session")
    args = parser.parse_args(argv)

    if args.wait_for_tmux_session:
        while (
            subprocess.run(
                ["tmux", "has-session", "-t", args.wait_for_tmux_session],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        ):
            print(
                f"Waiting for tmux session {args.wait_for_tmux_session} to finish",
                flush=True,
            )
            time.sleep(60)

    executable = _install(args.venv_dir.expanduser().resolve(), args.python)
    _verify(executable)
    print(f"Policy-enabled vLLM executable: {executable}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
