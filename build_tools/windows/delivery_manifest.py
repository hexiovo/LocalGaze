"""Write package hashes outside the archive so the ZIP does not hash itself."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


if __name__ == "__main__":
    replace_existing = len(sys.argv) == 5 and sys.argv[4] == "--replace"
    if len(sys.argv) not in (4, 5):
        raise SystemExit("usage: delivery_manifest.py RELEASE ZIP OUTPUT [--replace]")
    release, archive, output = map(Path, sys.argv[1:4])
    bundled_manifest = json.loads((release / "release-manifest.json").read_text(encoding="utf-8"))
    payload = {
        "release": release.name,
        "project_version": bundled_manifest.get("project_version"),
        "archive": archive.name,
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": digest(archive),
        "release_manifest_sha256": digest(release / "release-manifest.json"),
        "executables_sha256": {
            name: digest(release / name)
            for name in ("LocalGaze.exe", "LocalGazeWorker.exe", "ROIDrawing.exe")
        },
        "version_file_sha256": digest(release / "version.md"),
        "delivery_manifest_is_external": True,
    }
    mode = "w" if replace_existing else "x"
    with output.open(mode, encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
