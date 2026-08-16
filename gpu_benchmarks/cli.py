import argparse
import importlib
from collections.abc import Callable, Sequence

SUITES = {
    "attention": (
        "gpu_benchmarks.attention.cli",
        "Profile isolated scaled-dot-product attention kernels.",
    ),
    "model": (
        "gpu_benchmarks.model.cli",
        "Profile end-to-end Hugging Face model decode.",
    ),
    "components": (
        "gpu_benchmarks.components.cli",
        "Profile the staged synthetic decode ladder.",
    ),
    "prefix-cache": (
        "gpu_benchmarks.prefix_cache.cli",
        "Measure prefix-cache locality and workload homogeneity.",
    ),
    "vllm": (
        "gpu_benchmarks.serving.cli",
        "Profile vLLM serving and scheduling behavior.",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu-benchmarks",
        description="Benchmark memory behavior across LLM inference workloads.",
    )
    parser.add_argument("suite", nargs="?", choices=SUITES, help="Benchmark suite to run")
    parser.add_argument("suite_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.suite is None:
        parser.print_help()
        print("\nSuites:")
        for name, (_, description) in SUITES.items():
            print(f"  {name:<12} {description}")
        return 0

    module_name = SUITES[args.suite][0]
    suite_main: Callable[[list[str]], int] = importlib.import_module(module_name).main
    return suite_main(args.suite_args)
