from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pre import PreprocessConfig, preprocess_frame, preprocess_paths


class PreprocessingTests(unittest.TestCase):
    def test_vector_preprocessing_preserves_rows_and_marks_blink(self) -> None:
        t = pd.date_range("2026-01-01", periods=30, freq="40ms")
        frame = pd.DataFrame({
            "时间": t.astype(str),
            "x": np.linspace(100, 200, 30),
            "y": np.linspace(200, 300, 30),
            "blink": [0] * 12 + [1] + [0] * 17,
            "左瞳孔直径": np.linspace(0.2, 0.24, 30),
            "右瞳孔直径": np.linspace(0.21, 0.25, 30),
        })
        result = preprocess_frame(
            frame, time_column="时间", x_column="x", y_column="y",
            blink_column="blink", left_pupil_column="左瞳孔直径",
            right_pupil_column="右瞳孔直径",
            config=PreprocessConfig(coordinate_bounds=(0, 2559, 0, 1439)),
        )
        self.assertEqual(len(result.data), len(frame))
        self.assertGreaterEqual(result.data["Pre_BlinkPaddingMask"].sum(), 1)
        self.assertEqual(result.qc["coordinate_interpolation"], "disabled_no_model_calibration_distance")

    def test_datetime_resolution_and_pupil_independence_from_gaze_bounds(self) -> None:
        frame = pd.DataFrame({
            "Time": pd.date_range("2026-01-01", periods=20, freq="50ms"),
            "X": [100.0] * 8 + [3000.0] + [100.0] * 11,
            "Y": [100.0] * 20,
            "LeftPupil": np.linspace(0.20, 0.22, 20),
        })
        result = preprocess_frame(
            frame, time_column="Time", x_column="X", y_column="Y",
            left_pupil_column="LeftPupil",
            config=PreprocessConfig(coordinate_bounds=(0, 2559, 0, 1439)),
        )
        self.assertAlmostEqual(result.qc["duration_seconds"], 0.95)
        self.assertTrue(result.data.loc[8, "Pre_OutOfBoundsMask"])
        self.assertTrue(result.data.loc[8, "Pre_PupilValidMask"])

    def test_same_name_text_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            model_dir = tmp_path / "models"
            data_dir = tmp_path / "data"
            out_dir = tmp_path / "out"
            model_dir.mkdir()
            data_dir.mkdir()
            with pd.ExcelWriter(model_dir / "S01.xlsx") as writer:
                pd.DataFrame({"Error": [12.0, 18.0]}).to_excel(writer, sheet_name="NinePointCalibration", index=False)
            pd.DataFrame({
                "Timestamp": [f"2026-01-01 00:00:00.{i * 40:03d}" for i in range(10)],
                "X": np.arange(10) + 100.0, "Y": np.arange(10) + 200.0, "Blink": 0,
            }).to_csv(data_dir / "S01.tsv", sep="\t", index=False)
            report = preprocess_paths(model_dir, data_dir, out_dir)
            self.assertEqual(report.loc[0, "status"], "processed")
            output = pd.read_csv(out_dir / "S01.tsv", sep="\t")
            self.assertIn("Pre_X", output.columns)
            qc = json.loads((out_dir / "S01.tsv.qc.json").read_text(encoding="utf-8"))
            self.assertEqual(qc["calibration_error_pixels"], 15.0)


if __name__ == "__main__":
    unittest.main()
