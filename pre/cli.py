"""Command-line interface for LocalGaze preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

from .core import PreprocessConfig
from .pipeline import preprocess_mapping_table, preprocess_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="保持原文件格式完成眼动预处理；可使用模型目录、无模型数据或映射表。"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="数据文件或 Data 文件夹")
    source.add_argument("--mapping", help="CSV/XLSX 映射表；相对路径按表格所在目录解析")
    parser.add_argument("--model", help="可选模型文件或 Model_history 文件夹；省略时仅清理并执行 QC")
    parser.add_argument(
        "--output",
        help="输出目录；直接模式重复使用时需选择空目录，映射模式将其作为唯一运行目录的父目录",
    )
    parser.add_argument("--width", type=int, default=2560, help="屏幕宽度，默认 2560")
    parser.add_argument("--height", type=int, default=1440, help="屏幕高度，默认 1440")
    parser.add_argument(
        "--time-unit", choices=["seconds", "milliseconds", "microseconds"],
        default="seconds", help="仅用于纯数值时间列，默认 seconds",
    )
    parser.add_argument(
        "--preset", choices=["legacy", "analysis"], default="legacy",
        help="analysis 强制关闭 gaze XY 插值；默认 legacy 保留旧校准误差插值行为",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mapping and not args.output:
        parser.error("--mapping requires an explicit --output directory")
    config = PreprocessConfig(
        time_unit=args.time_unit,
        coordinate_bounds=(0.0, float(args.width - 1), 0.0, float(args.height - 1)),
    )
    if args.mapping:
        report = preprocess_mapping_table(args.mapping, args.output, config=config, preset=args.preset)
    else:
        report = preprocess_paths(args.model, args.data, args.output, config=config, preset=args.preset)
    failed = int((report["status"] == "failed").sum()) if not report.empty else 0
    invalid = int(report["status"].isin(["invalid_mapping_row"]).sum()) if not report.empty else 0
    duplicates = int((report["status"] == "duplicate_row").sum()) if not report.empty else 0
    processed = int((report["status"] == "processed").sum()) if not report.empty else 0
    print(f"Completed: processed={processed}, failed={failed}, invalid_rows={invalid}, duplicate_rows={duplicates}")
    if "summary_path" in report.attrs:
        print(f"Report: {report.attrs['summary_path']}")
    return 1 if failed or invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())

