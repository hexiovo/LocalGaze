"""Owned worker processes for the desktop view; importing starts no UI/camera."""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Mapping
import uuid

from session_names import canonical_session_id

PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_PREFIX = "LOCALGAZE_JOB_RESULT "
JOB_KINDS = {"calibrate", "capture", "known_target_offset", "roi", "preprocess", "analyze"}
CALIBRATION_MODEL_NAMES = {"ridge", "elastic_net", "linear_svr", "svr", "tiny_mlp"}
CALIBRATION_FORMULA_VERSIONS = {
    "eyetrax-landmark-features-v1",
    "eyetrax-compact-eye-head-v2",
}
DEFAULT_CALIBRATION_MODEL_NAME = "ridge"
DEFAULT_CALIBRATION_FORMULA_VERSION = "eyetrax-compact-eye-head-v2"


def _payload(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if kind not in JOB_KINDS:
        raise ValueError(f"unknown job kind: {kind}")
    if not isinstance(payload, Mapping):
        raise TypeError("job payload must be a mapping")
    allowed = {"source_path", "data_path", "model_path", "output_root", "time_unit", "session_id", "roi_path", "correction_path", "target_xy_px", "model_name", "feature_formula_version"}
    if set(payload) - allowed:
        raise ValueError("unsupported job payload fields: " + ", ".join(sorted(set(payload) - allowed)))
    if kind == "known_target_offset" and any(payload.get(name) is not None for name in ("source_path", "data_path")):
        raise ValueError("known_target_offset captures its own target window; external source paths are not accepted")
    result = dict(payload)
    if kind == "calibrate":
        model_name = result.get("model_name")
        formula_version = result.get("feature_formula_version")
        model_name = DEFAULT_CALIBRATION_MODEL_NAME if model_name is None else model_name
        formula_version = DEFAULT_CALIBRATION_FORMULA_VERSION if formula_version is None else formula_version
        if not isinstance(model_name, str) or model_name not in CALIBRATION_MODEL_NAMES:
            raise ValueError("calibrate model_name must be a supported estimator type")
        if not isinstance(formula_version, str) or formula_version not in CALIBRATION_FORMULA_VERSIONS:
            raise ValueError("calibrate feature_formula_version must be a supported formula version")
        result["model_name"] = model_name
        result["feature_formula_version"] = formula_version
    elif "model_name" in result or "feature_formula_version" in result:
        raise ValueError(f"{kind} does not accept calibration model/formula overrides")
    if result.get("session_id") is not None:
        result["session_id"] = canonical_session_id(result["session_id"])
    required = {"output_root"}
    if kind in {"calibrate", "capture", "known_target_offset"}:
        required.add("session_id")
    if kind in {"capture", "known_target_offset"}:
        required.add("model_path")
    if kind == "preprocess":
        required.add("model_path")
        if not result.get("source_path"):
            result["source_path"] = result.get("data_path")
        required.add("source_path")
    if kind == "analyze":
        required.add("source_path")
    for key in required:
        if not result.get(key):
            raise ValueError(f"{kind} requires {key}")
    for key in ("source_path", "data_path", "model_path", "output_root", "roi_path", "correction_path"):
        if result.get(key) is not None:
            result[key] = str(Path(result[key]).expanduser().resolve())
    if Path(result["output_root"]).exists() and not Path(result["output_root"]).is_dir():
        raise ValueError("output_root must be a directory")
    if result.get("time_unit") not in {None, "seconds", "milliseconds", "microseconds"}:
        raise ValueError("time_unit must be seconds, milliseconds or microseconds")
    if kind == "preprocess" and result.get("time_unit") is None:
        raise ValueError("preprocess requires an explicit numeric time_unit; use seconds for elapsed_s")
    if kind in {"capture", "known_target_offset"}:
        model = Path(result["model_path"])
        if not model.is_file() or model.suffix.lower() != ".pkl":
            raise ValueError(f"{kind} model_path must be an existing .pkl model")
    if kind == "known_target_offset":
        if result.get("correction_path") is None:
            result["correction_path"] = str(Path(result["output_root"]) / f"{result['session_id']}.offset-correction.json")
        correction = Path(result["correction_path"])
        if correction.suffix.lower() != ".json":
            raise ValueError("known_target_offset correction_path must end in .json")
        if correction.exists():
            raise FileExistsError("correction_path already exists; choose a new path")
        target = result.get("target_xy_px")
        if target is not None:
            if not isinstance(target, (list, tuple)) or len(target) != 2:
                raise ValueError("target_xy_px must contain two coordinates")
            normalized = []
            for coordinate in target:
                if isinstance(coordinate, bool):
                    raise ValueError("target_xy_px must contain finite numbers")
                try:
                    value = float(coordinate)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("target_xy_px must contain finite numbers") from exc
                if not math.isfinite(value):
                    raise ValueError("target_xy_px must contain finite numbers")
                normalized.append(value)
            result["target_xy_px"] = normalized
    if kind in {"calibrate", "capture", "known_target_offset"}:
        directory = Path(result["output_root"])
        suffixes = (".pkl", ".xlsx") if kind == "calibrate" else (".txt", ".xlsx", ".manifest.json", ".offset-correction.json") if kind == "known_target_offset" else (".txt", ".xlsx", ".manifest.json")
        if any((directory / (result["session_id"] + suffix)).exists() for suffix in suffixes):
            raise FileExistsError("session outputs already exist; choose a new session_id")
    json.dumps(result, allow_nan=False)
    return result


class JobManager:
    """One explicitly started owned process; all methods are nonblocking.

    The Tk thread calls start/poll/cancel/shutdown. A private reader thread
    handles pipe I/O only; it never calls Tk or controls another process.
    """
    def __init__(self, python_executable: str | None = None):
        self.python_executable = str(python_executable or sys.executable)
        self.state = "idle"
        self.job_id: str | None = None
        self._process = None
        self._reader_done = threading.Event()
        self._reader_done.set()
        self._messages = deque(maxlen=512)
        self._lock = threading.Lock()
        self._result = None
        self._dropped_logs = 0
        self._cancel_requested = False
        self._cancel_at = None
        self._closing = False
        self._pending = []

    @property
    def busy(self) -> bool:
        return self._process is not None

    @property
    def closed(self) -> bool:
        return self._closing and not self.busy and self._reader_done.is_set()

    def start(self, kind: str, payload: Mapping[str, Any]) -> str:
        if self._closing:
            raise ValueError("job manager is closing")
        if self.busy:
            raise ValueError("an owned job is already running")
        prepared = _payload(kind, payload)
        job_id = uuid.uuid4().hex
        arguments = ["--worker", kind, "--job-id", job_id, "--payload-json", json.dumps(prepared, ensure_ascii=False)]
        if getattr(sys, "frozen", False):
            worker = Path(sys.executable).resolve().parent / "LocalGazeWorker.exe"
            if not worker.is_file():
                raise FileNotFoundError(f"packaged worker is missing: {worker}; keep the complete application folder")
            command = [str(worker), *arguments]
        else:
            command = [self.python_executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), *arguments]
        env = os.environ.copy()
        env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
        process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace",
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
        self._process, self.job_id = process, job_id
        self.state = "running"
        self._cancel_requested = False
        self._cancel_at = None
        with self._lock:
            self._messages.clear()
            self._result = None
            self._dropped_logs = 0
        self._reader_done.clear()
        self._pending = [dict(job_id=job_id, type="started", message=f"{kind} started", returncode=None, outputs={}, details={"kind": kind, "session_id": prepared.get("session_id"), "output_root": prepared["output_root"]})]
        threading.Thread(target=self._read, args=(process, job_id), name="desktop-job-pipe", daemon=True).start()
        return job_id

    def _read(self, process, job_id):
        try:
            for line in process.stdout:
                text = line.rstrip("\r\n")
                result = None
                if text.startswith(RESULT_PREFIX):
                    try:
                        candidate = json.loads(text[len(RESULT_PREFIX):])
                        if isinstance(candidate, dict) and candidate.get("job_id") == job_id:
                            result = candidate
                    except (TypeError, ValueError):
                        pass
                with self._lock:
                    if job_id != self.job_id:
                        continue
                    if result is not None:
                        self._result = result
                    else:
                        if len(self._messages) == self._messages.maxlen:
                            self._dropped_logs += 1
                        self._messages.append(dict(job_id=job_id, type="log", message=text[:8192], returncode=None, outputs={}, details={}))
        except (OSError, ValueError) as exc:
            with self._lock:
                self._messages.append(dict(job_id=job_id, type="log", message=f"worker pipe: {exc}", returncode=None, outputs={}, details={}))
        finally:
            if process.stdout is not None:
                process.stdout.close()
            self._reader_done.set()

    def poll(self) -> list[dict[str, Any]]:
        current = self.job_id
        events, self._pending = self._pending, []
        with self._lock:
            events.extend(self._messages)
            self._messages.clear()
            if self._dropped_logs:
                events.append(dict(job_id=current, type="log", message=f"{self._dropped_logs} older log lines omitted", returncode=None, outputs={}, details={"truncated_log_lines": self._dropped_logs}))
                self._dropped_logs = 0
            result = self._result
        process = self._process
        if process is not None:
            returncode = process.poll()
            if returncode is None and self._cancel_requested and self._cancel_at is not None and time.monotonic() - self._cancel_at >= 2.:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            if returncode is not None and self._reader_done.is_set():
                # The pipe reader can finish after the first lock was released.
                # Re-read its final result only after observing its done event.
                with self._lock:
                    result = self._result
                outputs = result.get("outputs", {}) if isinstance(result, dict) else {}
                details = result.get("details", {}) if isinstance(result, dict) else {}
                if not isinstance(outputs, dict):
                    outputs = {}
                if not isinstance(details, dict):
                    details = {}
                if self._cancel_requested:
                    terminal, message = "cancelled", "owned worker exited after cancellation; any partial outputs are retained"
                    details = {**details, "partial_outputs_possible": True}
                elif returncode != 0:
                    terminal, message = "failed", result.get("message", f"worker exited with code {returncode}") if isinstance(result, dict) else f"worker exited with code {returncode}"
                elif not isinstance(result, dict) or result.get("status") not in {"completed", "failed", "cancelled"}:
                    terminal, message = "failed", "worker exited without a valid final result"
                else:
                    terminal = result["status"]
                    message = result.get("message", terminal)
                self.state = terminal
                self._process = None
                events.append(dict(job_id=current, type=terminal, message=message, returncode=returncode, outputs=outputs, details=details))
        return [event for event in events if event.get("job_id") == current]

    def cancel(self) -> bool:
        if self._process is None or self._process.poll() is not None:
            return False
        if self._cancel_requested:
            return True
        self._cancel_requested = True
        self._cancel_at = time.monotonic()
        self.state = "cancelling"
        try:
            self._process.terminate()
        except ProcessLookupError:
            pass
        return True

    def shutdown(self) -> None:
        self._closing = True
        self.cancel()


def _roi_job(payload):
    """Track writes performed by this app, rather than scanning others' files."""
    # Package-qualified imports keep the ROI globals separate from camera globals.
    from ROIDrawing.roi_drawer import ROIApp
    import tkinter as tk
    root = tk.Tk()
    outputs = {}
    try:
        app = ROIApp(root)
        original_writer = app._write_workbook_exclusive
        def write_owned(path, first, metadata):
            original_writer(path, first, metadata)
            outputs["roi_workbook"] = str(Path(path).resolve())
        app._write_workbook_exclusive = write_owned
        # The old save callback accepts its base directory from sys.argv.
        sys.argv = [sys.argv[0], payload["output_root"]]
        root.mainloop()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass
    return dict(status="completed" if outputs else "cancelled", outputs=outputs, details={"reason": "roi_saved" if outputs else "roi_closed_without_save"})


def run_worker_task(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Lazy API dispatch; called only inside a deliberately started worker."""
    request = _payload(kind, payload)
    if kind in {"calibrate", "capture"}:
        sys.path.insert(0, str(PROJECT_ROOT / "LocalGaze"))
        from eyetracking_pipeline import run_eye_tracking_workflow
        return run_eye_tracking_workflow(
            session_id=request["session_id"], save_dir=request["output_root"],
            workflow_mode=kind, model_path=request.get("model_path"),
            model_name=request.get("model_name") if kind == "calibrate" else None,
            feature_formula_version=request.get("feature_formula_version") if kind == "calibrate" else None,
            correction_path=request.get("correction_path") if kind == "capture" else None,
        )
    if kind == "known_target_offset":
        sys.path.insert(0, str(PROJECT_ROOT / "LocalGaze"))
        from eyetracking_pipeline import run_known_target_offset_workflow
        return run_known_target_offset_workflow(
            session_id=request["session_id"], save_dir=request["output_root"],
            model_path=request["model_path"], correction_path=request["correction_path"],
            target_xy_px=request.get("target_xy_px"),
        )
    if kind == "roi":
        return _roi_job(request)
    if kind == "preprocess":
        from pre import preprocess_analysis_paths, PreprocessConfig
        cfg = PreprocessConfig(time_unit=request["time_unit"])
        report = preprocess_analysis_paths(request["model_path"], request["source_path"], request["output_root"], config=cfg)
        counts = report["status"].value_counts().to_dict() if not report.empty else {}
        failed = report.empty or any(key != "processed" for key in counts)
        processed_paths = []
        if not report.empty and "output_file" in report.columns:
            processed_paths = [
                str(Path(value).resolve())
                for value in report.loc[report["status"] == "processed", "output_file"].dropna().tolist()
                if Path(value).is_file()
            ]
        outputs = {
            "directory": request["output_root"],
            "report": str(Path(request["output_root"]) / "preprocess_report.csv"),
        }
        if len(processed_paths) == 1:
            outputs["processed_file"] = processed_paths[0]
        return dict(status="failed" if failed else "completed",
                    outputs=outputs,
                    details={"status_counts": counts, "partial_outputs": failed, "source_path": request["source_path"], "output_root": request["output_root"], "session_id": request.get("session_id")})
    from EyeEvaluate.Evaluate.analysis_pipeline import analyze_file
    run = analyze_file(request["source_path"], request["output_root"], roi_path=request.get("roi_path"), time_unit=request.get("time_unit"), session_id=request.get("session_id"))
    return dict(status="completed", outputs={key: str(path) for key, path in run.outputs.items()}, details={"source_path": request["source_path"], "output_root": request["output_root"], "session_id": request.get("session_id"), "source_session_id": run.metadata.get("session_id"), "analysis_method_version": run.result.methods.get("method_version")})


def main(argv=None) -> int:
    # Frozen Python does not necessarily honor PYTHONIOENCODING.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Internal owned desktop worker")
    parser.add_argument("--worker", required=True, choices=sorted(JOB_KINDS))
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--payload-json", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_worker_task(args.worker, json.loads(args.payload_json))
        if not isinstance(result, dict) or result.get("status") not in {"completed", "failed", "cancelled"}:
            raise ValueError("backend did not return a valid outcome")
        if result["status"] == "completed":
            outputs = result.get("outputs")
            if not isinstance(outputs, dict) or not outputs or any(not Path(path).exists() for path in outputs.values()):
                raise ValueError("backend completed without all reported output paths")
    except Exception as exc:
        result = dict(status="failed", message=f"{type(exc).__name__}: {exc}", outputs={}, details={"error_type": type(exc).__name__, "partial_outputs_possible": True})
    result["job_id"] = args.job_id
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["JobManager", "run_worker_task", "JOB_KINDS"]
