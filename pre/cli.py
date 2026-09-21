"""Command-line interface for LocalGaze preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

from .core import PreprocessConfig
from .pipeline import preprocess_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按同名规则匹配 LocalGaze 模型与数据，并保持原文件格式完成眼动预处理。"
    )
    parser.add_argument("--model", required=True, help="模型文件或 Model_history 文件夹")
    parser.add_argument("--data", required=True, help="数据文件或 Data 文件夹")
    parser.add_argument("--output", help="输出文件夹；省略时在数据文件夹旁创建 *_preprocessed")
    parser.add_argument("--width", type=int, default=2560, help="屏幕宽度，默认 2560")
    parser.add_argument("--height", type=int, default=1440, help="屏幕高度，默认 1440")
    parser.add_argument(
        "--time-unit", choices=["seconds", "milliseconds", "microseconds"],
        default="seconds", help="仅用于纯数值时间列，默认 seconds",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PreprocessConfig(
        time_unit=args.time_unit,
        coordinate_bounds=(0.0, float(args.width - 1), 0.0, float(args.height - 1)),
    )
    report = preprocess_paths(args.model, args.data, args.output, config=config)
    failed = int((report["status"] == "failed").sum()) if not report.empty else 0
    unmatched = int(report["status"].isin(["data_without_model", "model_without_data"]).sum()) if not report.empty else 0
    processed = int((report["status"] == "processed").sum()) if not report.empty else 0
    print(f"Completed: processed={processed}, failed={failed}, unmatched={unmatched}")
    return 1 if failed or unmatched else 0


if __name__ == "__main__":
    raise SystemExit(main())

