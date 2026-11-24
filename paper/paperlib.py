"""Shared plumbing for the paper scripts: run directories, machine info, percentiles.

Everything about the corpus, tokenizers, vocabularies and the connection lives in
benchmarks/pgcommon.py and is imported from there, so the paper scripts and the README
benchmarks load identical data the identical way.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "benchmarks"))

import pgcommon as C  # noqa: E402,F401  (re-exported for the scripts)


def git_sha() -> str:
    """Commit of the checkout, `-dirty` if it has changes. Docker images have no .git, so they
    pass the host's SHA in PGTOKEN_GIT_SHA at build time."""
    env = os.environ.get("PGTOKEN_GIT_SHA")
    if env:
        return env
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=REPO, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               cwd=REPO).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def machine() -> dict:
    cpu = "?"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    mem_gb = None
    try:
        with open("/proc/meminfo") as f:
            mem_gb = round(int(f.readline().split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    return {
        "cpu": cpu,
        "cores": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "mem_gb": mem_gb,
        "host": os.environ.get("PGTOKEN_HOST_NAME", platform.node()),
        "python": platform.python_version(),
        "loadavg": os.getloadavg(),
        "git": git_sha(),
    }


def run_dir(name: str, smoke: bool, root: str | None = None) -> str:
    """runs/<date>-<name>/ (or runs/smoke/<date>-<name>/). Refuses to reuse an existing one."""
    base = root or os.path.join(HERE, "runs")
    if smoke:
        base = os.path.join(base, "smoke")
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(base, f"{stamp}-{name}")
    os.makedirs(path, exist_ok=False)
    return path


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_np_default)


def _np_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def pct(a, p) -> float:
    a = np.asarray(a, dtype=np.float64)
    return float(np.percentile(a, p)) if a.size else float("nan")


def dist(a) -> dict:
    a = np.asarray(a, dtype=np.float64)
    return {"n": int(a.size), "p50": pct(a, 50), "p90": pct(a, 90), "p99": pct(a, 99),
            "mean": float(a.mean()) if a.size else float("nan")}
