#!/usr/bin/env python3
"""Check a desktop preset against LocalGaze's real validator (schema only)."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
ALLOWED = {"known", "proposed", "missing", "not_applicable"}
REQUIRED = {"section", "parameter", "value", "unit", "source", "status", "risk"}

def pairs(items):
    result = {}
    for key, value in items:
        if key in result: raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result

def constant(value): raise ValueError(f"non-finite JSON value is forbidden: {value}")

def ledger(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED.issubset(set(reader.fieldnames or [])):
            raise ValueError("ledger missing required columns: " + ", ".join(sorted(REQUIRED-set(reader.fieldnames or []))))
        seen = set(); count = 0
        for line, row in enumerate(reader, 2):
            count += 1; key = (row["section"].strip(), row["parameter"].strip())
            if not all(key) or key in seen: raise ValueError(f"ledger row {line}: blank or duplicate parameter")
            seen.add(key)
            status = row["status"].strip()
            if status not in ALLOWED: raise ValueError(f"ledger row {line}: invalid status {status!r}")
            if any(not row[field].strip() for field in ("unit", "source", "risk")):
                raise ValueError(f"ledger row {line}: unit, source and risk are required")
            value = row["value"].strip()
            if status in {"known", "proposed"} and not value: raise ValueError(f"ledger row {line}: {status} requires value")
            if status == "missing" and value: raise ValueError(f"ledger row {line}: missing value must be empty")
        if not count: raise ValueError("ledger has no rows")
    return count

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path); p.add_argument("--project-root", required=True, type=Path)
    p.add_argument("--ledger", type=Path); a = p.parse_args()
    try:
        root = a.project_root.resolve(strict=True); cfg = a.config.resolve(strict=True)
        if not (root / "desktop_configuration.py").is_file(): raise ValueError("project root lacks desktop_configuration.py")
        sys.path.insert(0, str(root))
        from desktop_configuration import _validate_document
        doc = json.loads(cfg.read_text(encoding="utf-8-sig"), object_pairs_hook=pairs, parse_constant=constant)
        _validate_document(doc)
        n = ledger(a.ledger.resolve(strict=True)) if a.ledger else None
    except Exception as e:
        print(f"REJECTED: {type(e).__name__}: {e}", file=sys.stderr); return 2
    print("STRUCTURE_LOADABLE: accepted by target _validate_document")
    print("No GUI, data, scientific, pilot, or execution result is implied.")
    if n is not None: print(f"LEDGER_SHAPE_OK: {n} rows; labels are not validation evidence")
    return 0
if __name__ == "__main__": raise SystemExit(main())
