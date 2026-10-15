"""Opt-in, camera-free installation diagnostics used for Windows delivery."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback


def _wait_job(manager, kind, payload):
    job_id = manager.start(kind, payload)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        for event in manager.poll():
            if event["type"] in {"completed", "failed", "cancelled"}:
                if event["job_id"] != job_id or event["type"] != "completed":
                    raise RuntimeError(str(event))
                if not event["outputs"] or not all(Path(p).exists() for p in event["outputs"].values()):
                    raise RuntimeError("worker reported missing outputs")
                return event
        time.sleep(.1)
    manager.shutdown()
    close_deadline = time.monotonic() + 10
    while not manager.closed and time.monotonic() < close_deadline:
        manager.poll()
        time.sleep(.1)
    raise TimeoutError(f"{kind} worker exceeded the diagnostic time limit")


def check_installation(report_path: Path) -> int:
    """Write a new diagnostic report; never open a camera or read user data."""
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve before starting work; a repeat cannot overwrite prior evidence.
    with report_path.open("x", encoding="utf-8") as report_stream:
        results = []
        def check(name, action):
            try:
                details = action()
                results.append({"name": name, "status": "passed", "details": details})
            except Exception as exc:
                results.append({"name": name, "status": "failed", "error": str(exc),
                                "traceback": traceback.format_exc()})

        component = Path(__file__).resolve().parent / "LocalGaze"
        if str(component) not in sys.path:
            sys.path.insert(0, str(component))
        with tempfile.TemporaryDirectory(prefix="localgaze-check-", dir=report_path.parent) as directory:
            work = Path(directory)

            def backend_imports():
                for name in ("eyetracking_pipeline", "ROIDrawing.roi_drawer",
                             "EyeEvaluate.Evaluate.analysis_pipeline", "EyeEvaluate.Evaluate.analysis_options",
                             "pre.pipeline",
                             "eyetrax.features", "eyetrax.stability", "eyetrax.offset",
                             "eyetrax.utils.resources", "eyetrax._version"):
                    importlib.import_module(name)
                from Global_data import ALL_MODELS
                from ROIDrawing.Global_data import TRANSPARENCY
                assert "ridge" in ALL_MODELS and TRANSPARENCY == .2
                return {"modules": ["capture", "ROI", "pre", "analysis", "features", "stability", "offset", "resource adapter"],
                        "camera_opened": False}
            check("lazy_backend_imports_and_separate_globals", backend_imports)

            def durable_capture_and_brand_assets():
                import inspect
                import capture_persistence
                import capture_export_recovery
                import job_logging
                import eyetrax.movement
                import eyetrax.movement_runtime
                from PIL import Image
                guide_assets = sorted((Path(__file__).resolve().parent / "assets" / "guide").glob("*.png"))
                expected_guide_assets = [
                    "01-workbench.png", "02-analysis-selection.png", "03-analysis-parameters.png",
                    "04-known-target.png", "05-environment-check.png",
                ]
                assert [path.name for path in guide_assets] == expected_guide_assets
                for path in guide_assets:
                    with Image.open(path) as image:
                        image.verify()
                assert "output_root" in inspect.signature(capture_persistence.cleanup_capture_sidecars).parameters
                from eyetracking_pipeline import retry_capture_from_context, retry_capture_export
                assert callable(retry_capture_from_context)
                assert callable(retry_capture_export)
                asset_root = Path(__file__).resolve().parent / "assets"
                with Image.open(asset_root / "localgaze.ico") as icon:
                    sizes = sorted(icon.ico.sizes())
                assert sizes == [(s, s) for s in (16, 20, 24, 30, 32, 36, 40, 44, 48, 64, 96, 128, 256)]
                return {"durable_modules": "imported", "verified_root_cleanup": True,
                        "recovery_entry_points": "available", "icon_sizes": sizes,
                        "guide_screenshots": [path.name for path in guide_assets]}
            check("durable_capture_modules_and_brand_assets", durable_capture_and_brand_assets)

            def versioned_algorithm_modules():
                from eyetrax import __version__
                from eyetrax.features import DEFAULT_FEATURE_FORMULA_VERSION, FEATURE_FORMULA_V2, get_feature_schema
                from eyetrax.offset import OFFSET_SCHEMA
                from eyetrax.stability import CalibrationStabilityConfig
                schema = get_feature_schema(DEFAULT_FEATURE_FORMULA_VERSION)
                assert __version__ == "0.3.1"
                assert DEFAULT_FEATURE_FORMULA_VERSION == FEATURE_FORMULA_V2
                assert schema["feature_count"] == 20
                assert OFFSET_SCHEMA == "LocalGaze.offset-correction/1"
                config = CalibrationStabilityConfig()
                assert config.min_window_frames >= 2 and config.min_window_span_ns > 0
                return {"eyetrax_version": __version__,
                        "default_feature_formula": DEFAULT_FEATURE_FORMULA_VERSION,
                        "feature_count": schema["feature_count"], "offset_schema": OFFSET_SCHEMA,
                        "stability_min_window_frames": config.min_window_frames}
            check("bundled_algorithm_modules_and_version_metadata", versioned_algorithm_modules)

            def models():
                import numpy as np
                from eyetrax.models import create_model, list_available_models
                from eyetrax.gaze import GazeEstimator
                names = list_available_models()
                assert set(names) == {"ridge", "elastic_net", "svr", "linear_svr", "tiny_mlp"}
                for name in names:
                    create_model(name)
                estimator = GazeEstimator(initialize_face_mesh=False)
                count = estimator.feature_schema["feature_count"]
                rng = np.random.default_rng(42)
                x = rng.normal(size=(30, count)).astype(np.float32)
                y = np.column_stack((x[:, 0] * 10 + 100, x[:, 1] * 10 + 200))
                estimator.train(x, y)
                before = estimator.predict(x[:3])
                model_path = work / "diagnostic.pkl"
                estimator.save_model(model_path)
                loaded = GazeEstimator(initialize_face_mesh=False)
                loaded.load_model(model_path)
                assert np.allclose(before, loaded.predict(x[:3]))
                return {"models": names, "feature_formula_version": estimator.feature_formula_version,
                        "feature_count": count, "save_load_predictions_match": True}
            check("model_factory_fit_and_versioned_roundtrip", models)

            def face_graph():
                import numpy as np
                from eyetrax.gaze import GazeEstimator
                from eyetrax.utils.resources import resource_cache_active
                estimator = GazeEstimator()
                try:
                    features, _ = estimator.extract_features(np.zeros((240, 320, 3), dtype=np.uint8))
                    assert features is None
                finally:
                    estimator.face_mesh.close()
                return {"bundled_graph_and_models": "loaded", "input": "synthetic black image",
                        "camera_opened": False, "unicode_resource_cache": resource_cache_active()}
            check("mediapipe_facemesh_graph_and_native_libraries", face_graph)

            def hidden_views():
                import tkinter as tk
                from desktop_app import DesktopApp, _enable_dpi_awareness
                from ROIDrawing.roi_drawer import ROIApp
                _enable_dpi_awareness()
                root = tk.Tk()
                root.withdraw()
                try:
                    app = DesktopApp(root, auto_poll=False)
                    root.update_idletasks()
                    assert len(app.notebook.tabs()) == 5
                    guide = app.guide_path.read_text(encoding="utf-8")
                    rendered_guide = app.guide_text.get("1.0", "end-1c")
                    guide_image_count = len(tuple(app.guide_image_dir.glob("*.png")))
                    assert rendered_guide.startswith(guide)
                    assert app.guide_path.name == "README.md"
                    assert guide_image_count > 0
                    assert len(app.guide_text.image_names()) == guide_image_count
                    manual_path = app._user_manual_path()
                    assert manual_path.is_file() and manual_path.read_bytes().startswith(b"%PDF-")
                    opened_manual = []
                    app._open_guide_file = lambda path, label: opened_manual.append((path, label))
                    app.manual_button.invoke()
                    assert opened_manual == [(manual_path, "产品介绍与使用手册 PDF")]
                    assert app._manager is None
                finally:
                    root.destroy()
                root = tk.Tk()
                root.withdraw()
                try:
                    ROIApp(root)
                    root.update_idletasks()
                finally:
                    root.destroy()
                return {"guide_loaded": True, "guide": str(app.guide_path),
                        "product_manual": str(manual_path),
                        "guide_screenshots": guide_image_count,
                        "workbench_pages": 5, "ROI_view_constructed": True, "visible_window": False}
            check("hidden_tk_workbench_guide_and_roi", hidden_views)

            def roi_workbook():
                import pandas as pd
                from ROIDrawing.roi_drawer import ROIApp
                from roi_contract import ROI, roi_metadata_record, format_legacy_description
                from EyeEvaluate.FileOperate.ROIextract import extract_ROI
                roi = ROI("diagnostic", "检查区域", "rectangle", {"left": 0, "top": 0, "right": 300, "bottom": 300})
                path = work / "roi.xlsx"
                rows = [{"命名": roi.name, "形状": roi.shape,
                         "描述": format_legacy_description(roi.shape, roi.geometry, roi.boundary_mode)}]
                ROIApp._write_workbook_exclusive(path, rows, [roi_metadata_record(roi, 0)])
                assert len(extract_ROI(path)) == 1
                with pd.ExcelFile(path) as workbook:
                    assert set(workbook.sheet_names) == {"ROI", "ROI_Metadata"}
                try:
                    ROIApp._write_workbook_exclusive(path, rows, [roi_metadata_record(roi, 0)])
                except FileExistsError:
                    pass
                else:
                    raise AssertionError("ROI output was overwritten")
                return {"two_sheet_roundtrip": True, "overwrite_blocked": True}
            check("roi_workbook_export_read_and_no_overwrite", roi_workbook)

            def owned_workers():
                import pandas as pd
                from desktop_jobs import JobManager
                model_dir, data_dir = work / "models", work / "data"
                model_dir.mkdir()
                data_dir.mkdir()
                with pd.ExcelWriter(model_dir / "检查.xlsx", engine="openpyxl") as writer:
                    pd.DataFrame({"Error": [3., 4.]}).to_excel(writer, sheet_name="NinePointCalibration", index=False)
                source = data_dir / "检查.tsv"
                pd.DataFrame({"time": [i * .04 for i in range(20)], "elapsed_s": [i * .04 for i in range(20)],
                              "x": [100 + i for i in range(20)], "y": [200 + i for i in range(20)],
                              "blink": [0] * 20, "sample_status": ["accepted"] * 20}).to_csv(source, sep="\t", index=False, encoding="utf-8")
                manager = JobManager()
                pre_result = _wait_job(manager, "preprocess", {"model_path": str(model_dir), "source_path": str(data_dir),
                                       "output_root": str(work / "pre-output"), "time_unit": "seconds", "session_id": "打包检查"})
                pre_output_root = Path(pre_result["outputs"]["directory"]).resolve()
                pre_table_path = Path(pre_result["outputs"]["processed_file"]).resolve()
                assert pre_table_path.is_file() and pre_table_path.parent == pre_output_root
                assert pre_table_path.name == "检查.tsv"
                pre_table = pd.read_csv(pre_table_path, sep="\t")
                assert len(pre_table) == 20 and "Pre_X" in pre_table
                analysis = _wait_job(manager, "analyze", {"source_path": str(source), "output_root": str(work / "analysis-output"),
                                     "session_id": "打包分析", "time_unit": None})
                outputs = analysis["outputs"]
                assert any(p.endswith(".xlsx") for p in outputs.values())
                assert any(p.endswith(".json") for p in outputs.values())
                assert any(p.endswith(".csv") for p in outputs.values())
                text_analysis = _wait_job(manager, "analyze", {
                    "source_path": str(source), "output_root": str(work / "analysis-txt-output"),
                    "session_id": "打包分析TXT", "time_unit": None,
                    "selected_metrics": ["trajectory", "fixation"],
                    "analysis_config": {"fixation_min_samples": 4}, "table_format": "txt",
                })
                txt_outputs = text_analysis["outputs"]
                metric_table = Path(txt_outputs["metrics"])
                metadata_path = Path(txt_outputs["metadata"])
                manifest_path = Path(txt_outputs["manifest"])
                assert metric_table.suffix == ".txt" and b"\t" in metric_table.read_bytes()
                metadata_text = metadata_path.read_text(encoding="utf-8-sig")
                assert "table_format: txt" in metadata_text
                assert "selected_metric_groups: trajectory, fixation" in metadata_text
                assert "fixation_min_samples: 4" in metadata_text
                assert manifest_path.parent.name == ".internal"
                txt_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                assert txt_manifest["analysis"]["table_format"] == "txt"
                assert txt_manifest["analysis"]["methods"]["selected_metric_groups"] == ["trajectory", "fixation"]
                # Exercise an asynchronous worker failure after the existing-file payload check.
                invalid_source = work / "无效分析输入.tsv"
                invalid_source.write_text("not_gaze_column\nvalue\n", encoding="utf-8")
                manager.start("analyze", {"source_path": str(invalid_source), "output_root": str(work / "failed")})
                deadline = time.monotonic() + 120
                while manager.busy and time.monotonic() < deadline:
                    manager.poll()
                    time.sleep(.1)
                assert not manager.busy and manager.state == "failed"
                manager.shutdown()
                assert manager.closed
                return {"preprocess": pre_result["type"], "analysis": analysis["type"],
                        "unicode_paths_and_pipe": True, "missing_input_failed": True,
                        "owned_children_exited": True, "export_files": sorted(outputs),
                        "tab_delimited_txt_and_selected_metrics": True}
            check("real_owned_workers_preprocess_and_unified_exports", owned_workers)

        result = {"status": "passed" if all(r["status"] == "passed" for r in results) else "failed",
                  "frozen": bool(getattr(sys, "frozen", False)), "executable": sys.executable,
                  "python": sys.version, "checks": results,
                  "limits": ["no camera", "synthetic data only", "hidden Tk only", "no independent gaze accuracy"]}
        json.dump(result, report_stream, ensure_ascii=False, indent=2)
    return 0 if result["status"] == "passed" else 1
