"""Archive the exact source of the Paper 1 pipeline: a tarball of the package, the experiment scripts, the frozen files, the manifests,
the tests and the pinned requirements, named by the code digest that every run records (code_sha256). The work tree is not committed,
and a hash identifies content without keeping it; this keeps a retrievable copy next to the results. Nothing is committed or pushed.

    python experiments/paper1/archive_source.py
"""

import hashlib
import io
import json
import tarfile
import time

from common import ROOT

from probe_validity.provenance import code_digest

OUT = ROOT / "experiments/paper1/archive"
PATTERNS = ("probe_validity/*.py", "experiments/paper1/*.py", "experiments/paper1/*.sh", "experiments/paper1/frozen/*", "experiments/paper1/manifests/*.json",
            "tests/probe_validity/*.py", "requirements-paper1.txt")


def files() -> list:
    return sorted({f for pattern in PATTERNS for f in ROOT.glob(pattern) if f.is_file()})


def archive() -> tuple:
    digest = code_digest(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"source_{digest[:12]}.tar.gz"
    if path.exists():
        return path, digest, False
    listing = {f.relative_to(ROOT).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in files()}
    manifest = json.dumps({"code_sha256": digest, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "files": listing}, indent=1).encode()
    with tarfile.open(path, "w:gz") as tar:
        for f in files():
            tar.add(f, arcname=f.relative_to(ROOT).as_posix())
        info = tarfile.TarInfo("ARCHIVE_MANIFEST.json")
        info.size = len(manifest)
        tar.addfile(info, io.BytesIO(manifest))
    return path, digest, True


if __name__ == "__main__":
    path, digest, created = archive()
    print(f"{'wrote' if created else 'already have'} {path.relative_to(ROOT)} (code_sha256 {digest[:16]}...)")
