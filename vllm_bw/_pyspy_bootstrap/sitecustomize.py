"""Permit the benchmark runner's py-spy process to attach under Yama.

Linux systems with ``kernel.yama.ptrace_scope=1`` only allow a process's
ancestors to trace it. The benchmark starts vLLM and py-spy as siblings, so
vLLM must opt in explicitly. This bootstrap is placed on PYTHONPATH only for
profiled server runs and is inherited by EngineCore worker processes.
"""

from __future__ import annotations

import ctypes
import os
import sys

_PR_SET_PTRACER = 0x59616D61
_PR_SET_PTRACER_ANY = ctypes.c_ulong(-1).value


def _allow_pyspy() -> None:
    if os.environ.get("VLLM_BW_ALLOW_PYSPY") != "1" or not sys.platform.startswith("linux"):
        return
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.prctl(_PR_SET_PTRACER, _PR_SET_PTRACER_ANY, 0, 0, 0)
    if result != 0:
        error = ctypes.get_errno()
        print(
            f"warning: could not enable py-spy attachment: {os.strerror(error)}",
            file=sys.stderr,
            flush=True,
        )


_allow_pyspy()
