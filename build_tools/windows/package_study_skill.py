"""Build portable skill documentation and self-contained release attachments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def safe_files(root: Path) -> list[Path]:
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        if "__pycache__" in path.relative_to(root).parts or path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if relative.is_absolute() or ".." in relative.parts or "\\" in relative.as_posix():
            raise ValueError(f"Unsafe package path: {relative}")
        result.append(path)
    return result


def markdown_block(path: Path, content: str) -> str:
    runs = re.findall(r"`+", content)
    fence = "`" * max(3, max((len(run) for run in runs), default=2) + 1)
    language = {".md": "markdown", ".py": "python", ".json": "json", ".csv": "csv", ".yaml": "yaml"}.get(path.suffix.lower(), "text")
    return f"## {path.as_posix()}\n\n{fence}{language}\n{content.rstrip()}\n{fence}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--portable-docs", type=Path, required=True)
    parser.add_argument("--attachments-root", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    source = project / "skills" / "localgaze-study-config"
    docs_source = project / "docs" / "skills"
    portable = args.portable_docs.resolve()
    attachments = args.attachments_root.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Skill source is missing: {source}")
    if portable.exists() and any(portable.iterdir()):
        raise FileExistsError(f"Portable skill docs already exist: {portable}")
    portable.mkdir(parents=True, exist_ok=True)
    attachments.mkdir(parents=True, exist_ok=True)

    skill_copy = portable / "localgaze-study-config"
    shutil.copytree(source, skill_copy, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    extra_names = ("README.md", "README.txt", "实验设计填写模板.md")
    for name in extra_names:
        source_file = docs_source / name
        if not source_file.is_file():
            raise FileNotFoundError(f"Skill guide is missing: {source_file}")
        shutil.copyfile(source_file, portable / name)

    version_text = (project / "version.md").read_text(encoding="utf-8-sig")
    match = re.search(r"Windows 程序版本：\s*`?([0-9]+(?:\.[0-9]+){3})", version_text)
    if match is None:
        raise ValueError("Cannot read the four-part Windows version from version.md")
    version = match.group(1)
    archive = attachments / f"LocalGaze-study-config-{version}.zip"
    paste = attachments / f"LocalGaze-study-config-{version}.md"
    manifest_path = attachments / "skill-package-manifest.json"
    files = safe_files(portable)
    combined = ["# LocalGaze 实验配置技能（完整单文件版）\n\n本文件包含随包说明、实验设计填写模板和技能目录中的全部文本资源。\n"]
    entries = []
    for path in files:
        portable_path = PurePosixPath(path.relative_to(portable).as_posix()).as_posix()
        archive_path = portable_path
        content = path.read_text(encoding="utf-8-sig")
        combined.append(markdown_block(Path(archive_path), content))
        entries.append({"portable_path": portable_path, "archive_path": archive_path,
                        "size_bytes": path.stat().st_size, "sha256": digest(path)})
    paste_content = "\n".join(combined).rstrip() + "\n"
    with paste.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(paste_content)

    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in files:
            portable_path = PurePosixPath(path.relative_to(portable).as_posix()).as_posix()
            archive_path = portable_path
            info = zipfile.ZipInfo(archive_path, (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        expected_names = {item["archive_path"] for item in entries}
        if bundle.testzip() is not None or names != expected_names or "localgaze-study-config/SKILL.md" not in names:
            raise RuntimeError("Skill ZIP failed its integrity check")

    manifest = {
        "schema_version": 1,
        "skill_name": "localgaze-study-config",
        "project_version": version,
        "archive": {"name": archive.name, "size_bytes": archive.stat().st_size, "sha256": digest(archive)},
        "single_file": {"name": paste.name, "size_bytes": paste.stat().st_size, "sha256": digest(paste)},
        "files": entries,
        "portable_docs_path": "docs/skills",
        "archive_root": ".",
        "validation": {"zip_integrity": "passed", "file_count": len(files)},
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "passed", "archive": archive.name, "single_file": paste.name,
                      "files": len(files), "archive_sha256": digest(archive)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
