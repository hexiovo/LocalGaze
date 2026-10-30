"""Write a path-free, allow-listed public manifest and a private source inventory."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import sys


EXCLUDED_SOURCE_PARTS = {
    ".git", ".venv", "venv", ".build-venv", "release", "build", "build-artifacts", "dist",
    "tmp", "output", "SampleData", ".idea", "__pycache__", ".pytest_cache", ".ruff_cache",
    "downloads", "deployment", "portable", "deliverables",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".log"}
DEPENDENCIES = (
    "pyinstaller", "mediapipe", "opencv-contrib-python", "numpy", "pandas",
    "scipy", "scikit-learn", "openpyxl", "pillow", "screeninfo", "xlrd",
)
VALIDATION_LIMITS = [
    "no real-camera test",
    "no independent gaze-accuracy measurement",
    "no clean-machine test",
    "physical multi-monitor DPI behavior has not been verified",
]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def safe_relative(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or "\\" in relative or ":" in relative:
        raise RuntimeError(f"Public file path is not a safe relative path: {relative}")
    return relative


def is_reparse_point(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return True
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def source_inventory(project: Path) -> dict[str, str]:
    result = {}
    for current, directories, files in os.walk(project, topdown=True, followlinks=False):
        base = Path(current)
        directories[:] = sorted(
            name for name in directories
            if name not in EXCLUDED_SOURCE_PARTS and not is_reparse_point(base / name)
        )
        for name in sorted(files):
            path = base / name
            if is_reparse_point(path) or path.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            result[path.relative_to(project).as_posix()] = digest(path)
    return result


def public_file_index(release: Path) -> list[dict[str, object]]:
    result = []
    for path in sorted(release.rglob("*")):
        if not path.is_file() or path.name == "release-manifest.json":
            continue
        result.append({
            "path": safe_relative(path, release),
            "size_bytes": path.stat().st_size,
            "sha256": digest(path),
        })
    return result


def parse_version_file(version_path: Path) -> str:
    text = version_path.read_text(encoding="utf-8-sig")
    match = re.search(r"Windows 程序版本：\s*`?([0-9]+(?:\.[0-9]+){2,3})", text)
    if match is None:
        raise RuntimeError("Cannot read the Windows program version from version.md")
    return match.group(1)


if __name__ == "__main__":
    replace_existing = len(sys.argv) == 6 and sys.argv[5] == "--replace-manifest"
    if len(sys.argv) not in (5, 6):
        raise SystemExit(
            "usage: release_manifest.py PROJECT RELEASE CHECK_REPORT SOURCE_INVENTORY "
            "[--replace-manifest]"
        )
    project, release, report, source_output = map(Path, sys.argv[1:5])
    project = project.resolve()
    release = release.resolve()
    if source_output.resolve().is_relative_to(release):
        raise RuntimeError("Private source inventory must stay outside the public release")

    check_result = json.loads(report.read_text(encoding="utf-8-sig"))
    checks = check_result.get("checks", [])
    passed_checks = sum(item.get("status") == "passed" for item in checks)
    if check_result.get("status") != "passed" or not checks or passed_checks != len(checks):
        raise RuntimeError("A passing installation report with all checks passed is required")

    algorithm_check = next((item for item in checks
                            if item.get("name") == "bundled_algorithm_modules_and_version_metadata"), {})
    algorithm_details = algorithm_check.get("details", {})
    metrics_source = (project / "EyeEvaluate" / "Evaluate" / "metrics.py").read_text(encoding="utf-8")
    method_match = re.search(r'^METHOD_VERSION\s*=\s*["\']([^"\']+)["\']$', metrics_source, re.MULTILINE)
    if method_match is None:
        raise RuntimeError("Cannot read the analysis method version from metrics.py")
    version_info = (project / "build_tools" / "windows" / "version_info.txt").read_text(encoding="utf-8")
    project_version = parse_version_file(project / "version.md")
    version_parts = tuple(int(part) for part in project_version.split("."))
    if len(version_parts) != 4:
        raise RuntimeError("The Windows program version must have four numeric components")
    expected_version = ".".join(str(part) for part in version_parts)
    windows_match = re.search(r"StringStruct\('FileVersion',\s*'([^']+)'\)", version_info)
    product_match = re.search(r"StringStruct\('ProductVersion',\s*'([^']+)'\)", version_info)
    if (windows_match is None or product_match is None or windows_match.group(1) != expected_version
            or product_match.group(1) != expected_version):
        raise RuntimeError("Windows file/product versions do not match version.md")

    source_hashes = source_inventory(project)
    source_doc = {"schema_version": 1, "files_sha256": source_hashes}
    source_output.parent.mkdir(parents=True, exist_ok=True)
    source_output.write_text(json.dumps(source_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    machine = platform.machine().lower()
    architecture = "x86_64" if machine in {"amd64", "x86_64"} else machine
    manifest = {
        "schema_version": 1,
        "release": release.name,
        "project_version": project_version,
        "platform": "windows",
        "architecture": architecture,
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "analysis_method_version": method_match.group(1),
        "eyetrax_version": algorithm_details.get("eyetrax_version"),
        "default_feature_formula": algorithm_details.get("default_feature_formula"),
        "version_file_sha256": digest(project / "version.md"),
        "dependencies": {name: importlib.metadata.version(name) for name in DEPENDENCIES},
        "files": public_file_index(release),
        "verification": {
            "status": "passed",
            "checks_passed": passed_checks,
            "checks_total": len(checks),
        },
        "validation_limits": VALIDATION_LIMITS,
    }
    output = release / "release-manifest.json"
    mode = "w" if replace_existing else "x"
    with output.open(mode, encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
