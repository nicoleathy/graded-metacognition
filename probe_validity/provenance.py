"""Hashes and repository state recorded with every result."""

import hashlib
import json
import subprocess
from pathlib import Path


def sha256_of(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def git_state(root: Path) -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", "probe_validity", "experiments"))}


def code_digest(root: Path) -> str:
    """SHA-256 over the source of the package and of the Paper 1 scripts (tests excluded): identifies the exact code
    of a run even when the working tree is not committed."""
    files = sorted([*root.glob("probe_validity/*.py"), *root.glob("experiments/paper1/*.py")])
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(root).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()
