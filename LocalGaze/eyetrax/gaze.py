from __future__ import annotations

import copy
import warnings
from collections import deque
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from typing import Any, Hashable, Sequence

import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    import mediapipe as mp
except ModuleNotFoundError:
    mp = None

from eyetrax.constants import LEFT_EYE_INDICES, MUTUAL_INDICES, RIGHT_EYE_INDICES
from eyetrax.data_contract import (
    DATA_CONTRACT_VERSION,
    IRIS_PROXY_FORMULA_VERSION,
    IrisProxyMetrics,
    compute_iris_proxy,
)
from eyetrax.models import BaseModel, create_model
from eyetrax.features import (
    FEATURE_FORMULA_V1, FEATURE_FORMULA_V2, DEFAULT_FEATURE_FORMULA_VERSION,
    get_feature_schema, head_descriptor_indices, extract_compact_features,
)


FEATURE_FORMULA_VERSION = "eyetrax-landmark-features-v1"
LEGACY_UNVERSIONED_FORMULA = "legacy-unversioned"
LEGACY_IRIS_FORMULA_VERSION = "legacy-iris-ring-477-472-v1"
CALIBRATION_LEDGER_VERSION = "target-visit-ledger-v1"
FIT_RESIDUAL_METHOD_VERSION = "calibration-fit-residuals-v1"
FEATURE_SCALE_DESCRIPTION = "StandardScaler followed by optional feature scaling"


def compute_pixel_ear(
    landmarks: Sequence[Any],
    image_width: int | float,
    image_height: int | float,
    *,
    inner_index: int,
    outer_index: int,
    upper_index: int,
    lower_index: int,
) -> float | None:
    """Compute eye aspect ratio after converting normalized x/y to pixels."""

    try:
        width, height = float(image_width), float(image_height)
        if not np.isfinite(width) or not np.isfinite(height) or width <= 0 or height <= 0:
            return None

        def point(index: int) -> np.ndarray:
            landmark = landmarks[index]
            if hasattr(landmark, "x") and hasattr(landmark, "y"):
                x, y = landmark.x, landmark.y
            else:
                x, y = landmark[0], landmark[1]
            xy = np.asarray([x, y], dtype=np.float64)
            xy *= np.asarray([width, height], dtype=np.float64)
            return xy

        horizontal = float(np.linalg.norm(point(outer_index) - point(inner_index)))
        vertical = float(np.linalg.norm(point(upper_index) - point(lower_index)))
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(horizontal) or not np.isfinite(vertical) or horizontal <= 0:
        return None
    ear = vertical / horizontal
    return float(ear) if np.isfinite(ear) else None


class EyeStateTracker:
    """Open-eye baseline with close/open hysteresis and explicit warm-up."""

    def __init__(
        self,
        *,
        history_len: int = 50,
        min_history: int = 15,
        close_ratio: float = 0.70,
        open_ratio: float = 0.85,
    ) -> None:
        if not 0 < close_ratio < open_ratio <= 1:
            raise ValueError("eye-state ratios must satisfy 0 < close < open <= 1")
        self._baseline: deque[float] = deque(maxlen=max(1, int(history_len)))
        self._warmup: list[float] = []
        self._min_history = max(1, int(min_history))
        self.close_ratio = float(close_ratio)
        self.open_ratio = float(open_ratio)
        self.state = "warmup"
        self._last_valid_state: str | None = None
        self.last_thresholds: dict[str, float | None] = {
            "close": None,
            "open": None,
        }

    @property
    def open_baseline(self) -> float | None:
        if not self._baseline:
            return None
        return float(np.mean(self._baseline))

    def update(self, ear: float | None) -> bool:
        """Return blink/unusable status; closed values never update baseline."""

        if ear is None or not np.isfinite(ear) or ear < 0:
            self.state = "invalid"
            return True

        value = float(ear)
        if len(self._warmup) < self._min_history:
            self._warmup.append(value)
            if len(self._warmup) == self._min_history:
                # Seed from the most open warm-up observations; thereafter only
                # hysteresis-confirmed open samples can change the baseline.
                ordered = sorted(self._warmup)
                top_count = max(1, int(np.ceil(len(ordered) * 0.30)))
                seed = float(np.mean(ordered[-top_count:]))
                self._baseline.append(seed)
            self.state = "warmup"
            self.last_thresholds = {"close": None, "open": None}
            return True

        baseline = self.open_baseline
        if baseline is None or not np.isfinite(baseline) or baseline <= 0:
            self.state = "warmup"
            return True

        close_threshold = baseline * self.close_ratio
        open_threshold = baseline * self.open_ratio
        self.last_thresholds = {"close": close_threshold, "open": open_threshold}

        if self._last_valid_state == "closed":
            is_open = value >= open_threshold
        else:
            is_open = value >= close_threshold

        if is_open:
            self.state = "open"
            self._last_valid_state = "open"
            if value >= open_threshold:
                self._baseline.append(value)
            return False

        self.state = "closed"
        self._last_valid_state = "closed"
        return True


def _software_versions() -> dict[str, str]:
    def version_or_module(distribution: str, module, attribute: str) -> str:
        try:
            return distribution_version(distribution)
        except PackageNotFoundError:
            return str(getattr(module, attribute, "unavailable"))

    versions = {
        "python": __import__("sys").version.split()[0],
        "numpy": str(np.__version__),
        "opencv": version_or_module("opencv-python", cv2, "__version__"),
        "mediapipe": version_or_module("mediapipe", mp, "__version__"),
    }
    for name in ("scikit-learn",):
        try:
            versions[name] = distribution_version(name)
        except PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def _feature_schema(formula_version=FEATURE_FORMULA_VERSION) -> dict[str, Any]:
    # Keep this historical helper's no-argument v1 meaning for old callers.
    return get_feature_schema(formula_version)


def _validate_versioned_feature_contract(model, expected_formula=None) -> None:
    """Reject versioned models whose recorded schema or feature scaling disagrees."""

    metadata = getattr(model, "metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("versioned model must contain dictionary metadata")
    formula = metadata.get("feature_formula_version")
    if formula not in {FEATURE_FORMULA_V1, FEATURE_FORMULA_V2} or (expected_formula is not None and formula != expected_formula):
        raise ValueError("versioned model feature formula does not match a supported active extractor")
    expected_schema = _feature_schema(formula)
    schema = metadata.get("feature_schema")
    if not isinstance(schema, dict):
        raise ValueError("versioned model is missing its feature_schema")
    schema_keys = tuple(expected_schema) if formula == FEATURE_FORMULA_V2 else (
        "schema_version", "feature_formula_version", "ordered_fields",
        "feature_count", "coordinate_source", "landmark_transform",
    )
    for name in schema_keys:
        if name not in schema or schema[name] != expected_schema[name]:
            raise ValueError(f"versioned model feature_schema.{name} is incompatible")
    if metadata.get("feature_scale_description") != FEATURE_SCALE_DESCRIPTION:
        raise ValueError("versioned model feature scaling description is incompatible")

    feature_count = expected_schema["feature_count"]
    native = getattr(model, "model", getattr(model, "_template", None))
    native_count = getattr(native, "n_features_in_", None)
    if native_count is not None and native_count != feature_count:
        raise ValueError("versioned model native feature count is incompatible")
    scaler = getattr(model, "scaler", None)
    scaler_feature_count = getattr(scaler, "n_features_in_", None)
    if scaler_feature_count is not None and scaler_feature_count != feature_count:
        raise ValueError("versioned model scaler feature count is incompatible with feature_schema")

    def require_feature_vector(values, name: str) -> np.ndarray:
        try:
            vector = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"versioned model {name} is not a numeric feature vector") from exc
        if vector.shape != (feature_count,) or not np.all(np.isfinite(vector)):
            raise ValueError(
                f"versioned model {name} must contain {feature_count} finite feature values"
            )
        return vector

    scaler_metadata = metadata.get("fitted_scaler")
    if scaler_metadata is not None:
        if not isinstance(scaler_metadata, dict):
            raise ValueError("versioned model fitted_scaler metadata must be a dictionary")
        if scaler_metadata.get("n_features_in") != feature_count:
            raise ValueError("versioned model fitted_scaler metadata feature count is incompatible")
        if scaler_feature_count is None:
            raise ValueError("versioned model fitted_scaler metadata has no matching fitted scaler")
        scaler_attributes = {"mean": "mean_", "scale": "scale_", "var": "var_"}
        for name, attribute in scaler_attributes.items():
            if name in scaler_metadata:
                recorded = require_feature_vector(scaler_metadata[name], f"fitted_scaler.{name}")
                actual_values = getattr(scaler, attribute, None)
                if actual_values is None or not np.array_equal(
                    recorded,
                    require_feature_vector(actual_values, f"scaler.{attribute}"),
                ):
                    raise ValueError(f"versioned model fitted_scaler.{name} metadata is inconsistent")

    if scaler is not None:
        for name in ("mean_", "scale_", "var_"):
            values = getattr(scaler, name, None)
            if values is not None:
                require_feature_vector(values, f"scaler.{name}")

    variable_scaling = getattr(model, "variable_scaling", None)
    if variable_scaling is not None:
        require_feature_vector(variable_scaling, "variable_scaling")
    training = metadata.get("training")
    if training is not None:
        if not isinstance(training, dict):
            raise ValueError("versioned model training metadata must be a dictionary")
        ledger = training.get("calibration_ledger", [])
        if not isinstance(ledger, list):
            raise ValueError("calibration ledger must be a list")
        for entry in ledger:
            if not isinstance(entry, dict):
                raise ValueError("calibration ledger entry must be a dictionary")
            entry_formula = entry.get("feature_formula_version", FEATURE_FORMULA_V1)
            if entry_formula != formula:
                raise ValueError("calibration ledger mixes feature formulas")
            require_feature_vector(entry.get("features"), "calibration ledger features")
        ranges = training.get("head_descriptor_training_range")
        if ranges is not None:
            values = np.asarray(ranges, dtype=float)
            if values.shape != (len(head_descriptor_indices(formula)), 2) or not np.isfinite(values).all() or np.any(values[:, 0] > values[:, 1]):
                raise ValueError("head descriptor training ranges are incompatible")
        if "variable_scaling" in training:
            recorded_scaling = training["variable_scaling"]
            if recorded_scaling is None:
                if variable_scaling is not None:
                    raise ValueError("versioned model variable_scaling metadata is inconsistent")
            else:
                recorded_vector = require_feature_vector(
                    recorded_scaling,
                    "training.variable_scaling",
                )
                if variable_scaling is None:
                    raise ValueError("versioned model is missing its recorded variable_scaling")
                actual_vector = require_feature_vector(variable_scaling, "variable_scaling")
                if not np.array_equal(recorded_vector, actual_vector):
                    raise ValueError("versioned model variable_scaling metadata is inconsistent")


class GazeEstimator:
    def __init__(
        self,
        model_name: str = "ridge",
        model_kwargs: dict | None = None,
        ear_history_len: int = 50,
        blink_threshold_ratio: float | None = None,
        min_history: int = 15,
        *,
        blink_close_ratio: float = 0.70,
        blink_open_ratio: float = 0.85,
        initialize_face_mesh: bool = True,
        feature_formula_version: str = DEFAULT_FEATURE_FORMULA_VERSION,
    ):
        if feature_formula_version not in {FEATURE_FORMULA_V1, FEATURE_FORMULA_V2}:
            raise ValueError("new estimators require a supported versioned feature formula")
        if blink_threshold_ratio is not None:
            # Keep the historical constructor argument as an explicit override.
            blink_close_ratio = float(blink_threshold_ratio)
            blink_open_ratio = max(float(blink_open_ratio), blink_close_ratio + 0.05)
        if initialize_face_mesh:
            if mp is None or cv2 is None:
                missing = [name for name, module in (("mediapipe", mp), ("opencv-python", cv2)) if module is None]
                raise ModuleNotFoundError(
                    "GazeEstimator face extraction requires installed dependencies: " + ", ".join(missing)
                )
            from eyetrax.utils.resources import prepare_face_mesh_resources
            prepare_face_mesh_resources()
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        else:
            self.face_mesh = None
        self.model_name = model_name
        self.model_kwargs = dict(model_kwargs or {})
        self.model: BaseModel = create_model(model_name, **self.model_kwargs)
        self.calibration_use_group_weights = True
        self.feature_formula_version = feature_formula_version
        self.feature_schema = _feature_schema(feature_formula_version)
        self._model_loaded = False
        self.last_feature_metadata: dict[str, Any] = {}
        self.screen_info: dict[str, int] | None = None
        self.last_runtime_flags: dict[str, Any] = {
            "head_descriptor_outside_training_range": None,
        }
        self.last_eye_state = "warmup"
        self.last_ear_metadata: dict[str, Any] = {}
        self.last_iris_measurement_metadata: dict[str, Any] = {}
        self._eye_state = EyeStateTracker(
            history_len=ear_history_len,
            min_history=min_history,
            close_ratio=blink_close_ratio,
            open_ratio=blink_open_ratio,
        )
        self._calibration_ledger: list[dict[str, Any]] = []
        self._seen_calibration_frames: set[tuple[str, str]] = set()
        self._next_visit_number = 1
        self._loaded_legacy_formula = False
        self._sync_feature_metadata()

    def _sync_feature_metadata(self) -> None:
        metadata = getattr(self.model, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.update(
            {
                "data_contract_version": DATA_CONTRACT_VERSION,
                "feature_formula_version": self.feature_formula_version,
                "feature_schema": copy.deepcopy(self.feature_schema),
                "feature_scale_description": FEATURE_SCALE_DESCRIPTION,
                "iris_proxy_formula_version": IRIS_PROXY_FORMULA_VERSION,
                "software_dependencies": _software_versions(),
                "screen_info_px": self.screen_info,
                "calibration_canvas_size_px": copy.deepcopy(self.screen_info),
                "canvas_reference_status": "dimensions_recorded_monitor_and_dpi_unverified" if self.screen_info else "unknown",
                "feature_contract_status": "versioned",
            }
        )
        self.model.metadata = metadata

    def set_screen_info(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("screen dimensions must be positive")
        current = {"width": int(width), "height": int(height)}
        self.current_canvas_info_px = current
        if self._model_loaded:
            # Current display dimensions cannot fill an unknown historical canvas.
            return
        self.screen_info = current
        if self.feature_formula_version in {FEATURE_FORMULA_V1, FEATURE_FORMULA_V2}:
            self._sync_feature_metadata()

    @staticmethod
    def _legacy_iris_values(landmarks, inter_eye_dist: float) -> tuple[float | None, float | None]:
        """Read the pre-versioned ring formula, retaining its original indices."""

        if not np.isfinite(inter_eye_dist) or inter_eye_dist <= 0:
            return None, None
        left_indices = [473, 474, 475, 476, 477]
        right_indices = [468, 469, 470, 471, 472]
        values = []
        for indices in (left_indices, right_indices):
            try:
                points = np.asarray(
                    [(landmarks[index].x, landmarks[index].y) for index in indices],
                    dtype=np.float64,
                )
            except (AttributeError, IndexError, TypeError, ValueError):
                values.append(None)
                continue
            center = points[-1]
            radii = np.linalg.norm(points[:-1] - center, axis=1)
            result = float((2 * np.mean(radii)) / inter_eye_dist)
            values.append(result if np.isfinite(result) else None)
        return values[0], values[1]

    def _set_head_descriptor_flags(self, features: np.ndarray | None) -> None:
        self.last_runtime_flags = {"head_descriptor_outside_training_range": None}
        if features is None:
            return
        metadata = getattr(self.model, "metadata", {})
        training = metadata.get("training", {}) if isinstance(metadata, dict) else {}
        ranges = training.get("head_descriptor_training_range") if isinstance(training, dict) else None
        indices = head_descriptor_indices(self.feature_formula_version)
        if not ranges or len(ranges) != len(indices):
            return
        descriptors = np.asarray(features, dtype=np.float64)[list(indices)]
        if not np.all(np.isfinite(descriptors)):
            return
        self.last_runtime_flags = {
            "head_descriptor_outside_training_range": any(
                value < bounds[0] or value > bounds[1]
                for value, bounds in zip(descriptors, ranges)
            )
        }

    def extract_features(
        self,
        image,
        return_more: bool = False,
        *,
        iris_formula_version: str = IRIS_PROXY_FORMULA_VERSION,
    ):
        """Extract gaze features and optional, explicitly versioned iris proxies.

        With ``return_more=True`` the default result is
        ``(features, blink, IrisProxyMetrics | None, metadata)``. Historical
        callers may explicitly request ``LEGACY_IRIS_FORMULA_VERSION``; that
        path returns the same four scalar tuple as the old API and records its
        legacy formula in ``last_iris_measurement_metadata``.
        """

        if self.face_mesh is None or cv2 is None:
            raise RuntimeError(
                "extract_features requires a FaceMesh initialized with OpenCV and MediaPipe"
            )

        if iris_formula_version not in {
            IRIS_PROXY_FORMULA_VERSION,
            LEGACY_IRIS_FORMULA_VERSION,
        }:
            raise ValueError(f"unsupported iris formula version: {iris_formula_version}")

        image_height, image_width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            self.last_eye_state = "no_face"
            self.last_feature_metadata = {"valid": False, "reason": "no_face", "stability_observables": None}
            self.last_runtime_flags = {"head_descriptor_outside_training_range": None}
            if return_more:
                self.last_iris_measurement_metadata = {
                    "formula_version": iris_formula_version,
                    "measurement_kind": "iris_ring_diameter_proxy",
                    "is_pupil_measurement": False,
                    "valid": False,
                    "reason": "no_face",
                }
                if iris_formula_version == LEGACY_IRIS_FORMULA_VERSION:
                    return None, False, None, None
                return None, False, None, dict(self.last_iris_measurement_metadata)
            return None, False

        landmarks = results.multi_face_landmarks[0].landmark
        compact = extract_compact_features(landmarks, image_width, image_height)
        self.last_feature_metadata = {
            "valid": compact.features is not None, "reason": compact.reason,
            "stability_observables": compact.stability_observables,
            "feature_formula_version": self.feature_formula_version,
        }
        if self.feature_formula_version == FEATURE_FORMULA_V2:
            features = compact.features
            # Only the explicitly requested historical iris proxy uses this
            # old normalized 3-D span; it is not a v2 feature normalizer.
            all_points = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)
            inter_eye_dist = np.linalg.norm(all_points[263] - all_points[33])
        else:
            all_points = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)
            nose_anchor = all_points[4]
            left_corner = all_points[33]
            right_corner = all_points[263]
            top_of_head = all_points[10]

            shifted_points = all_points - nose_anchor
            x_axis = right_corner - left_corner
            x_axis /= np.linalg.norm(x_axis) + 1e-9
            y_approx = top_of_head - nose_anchor
            y_approx /= np.linalg.norm(y_approx) + 1e-9
            y_axis = y_approx - np.dot(y_approx, x_axis) * x_axis
            y_axis /= np.linalg.norm(y_axis) + 1e-9
            z_axis = np.cross(x_axis, y_axis)
            z_axis /= np.linalg.norm(z_axis) + 1e-9
            rotation = np.column_stack((x_axis, y_axis, z_axis))
            rotated_points = (rotation.T @ shifted_points.T).T

            left_corner_rot = rotation.T @ (left_corner - nose_anchor)
            right_corner_rot = rotation.T @ (right_corner - nose_anchor)
            inter_eye_dist = np.linalg.norm(right_corner_rot - left_corner_rot)
            if inter_eye_dist > 1e-7:
                rotated_points /= inter_eye_dist

            subset_indices = LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MUTUAL_INDICES
            eye_landmarks = rotated_points[subset_indices]
            features = eye_landmarks.flatten()

            # These are coarse descriptors of the eye/head landmark basis. Their
            # values are features, not ground-truth camera yaw/pitch/roll angles.
            head_basis_descriptors = np.array(
                [
                    np.arctan2(rotation[1, 0], rotation[0, 0]),
                    np.arctan2(-rotation[2, 0], np.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2)),
                    np.arctan2(rotation[2, 1], rotation[2, 2]),
                ],
                dtype=np.float32,
            )
            features = np.concatenate([features, head_basis_descriptors])
        self._set_head_descriptor_flags(features)

        eye_specs = (
            (133, 33, 159, 145),
            (362, 263, 386, 374),
        )
        ears = [
            compute_pixel_ear(
                landmarks,
                image_width,
                image_height,
                inner_index=inner,
                outer_index=outer,
                upper_index=upper,
                lower_index=lower,
            )
            for inner, outer, upper, lower in eye_specs
        ]
        valid_ears = [value for value in ears if value is not None and np.isfinite(value)]
        ear = float(np.mean(valid_ears)) if len(valid_ears) == 2 else None
        blink_detected = self._eye_state.update(ear)
        self.last_eye_state = self._eye_state.state
        if self.feature_formula_version == FEATURE_FORMULA_V2 and features is None:
            self.last_eye_state = "invalid"
            blink_detected = True
        self.last_ear_metadata = {
            "left_ear": ears[0],
            "right_ear": ears[1],
            "pixel_dimensions": {"width": int(image_width), "height": int(image_height)},
            "open_baseline": self._eye_state.open_baseline,
            "thresholds": dict(self._eye_state.last_thresholds),
            "state": self.last_eye_state,
            "warmup_samples": len(self._eye_state._warmup),
            "baseline_samples": len(self._eye_state._baseline),
        }

        if not return_more:
            return features, blink_detected

        if iris_formula_version == LEGACY_IRIS_FORMULA_VERSION:
            left_legacy, right_legacy = self._legacy_iris_values(landmarks, inter_eye_dist)
            self.last_iris_measurement_metadata = {
                "formula_version": LEGACY_IRIS_FORMULA_VERSION,
                "measurement_kind": "legacy_iris_ring_ratio",
                "is_pupil_measurement": False,
                "valid": left_legacy is not None and right_legacy is not None,
                "legacy_center_indices": {"left": 477, "right": 472},
                "legacy_normalizer": "3-D normalized landmark eye-corner span",
                "warning": "historical iris-ring values were previously labeled as pupil diameter",
            }
            return features, blink_detected, left_legacy, right_legacy

        metrics = compute_iris_proxy(landmarks, image_width, image_height)
        self.last_iris_measurement_metadata = {
            "formula_version": IRIS_PROXY_FORMULA_VERSION,
            "measurement_kind": "iris_ring_diameter_proxy",
            "is_pupil_measurement": False,
            "valid": metrics is not None,
            "center_indices": {"left": 473, "right": 468},
            "ring_indices": {"left": [474, 475, 476, 477], "right": [469, 470, 471, 472]},
            "center_assignment_basis": "inferred from refined 478-point landmark layout",
            "coordinate_conversion": "x * image_width, y * image_height",
            "normalizer": "2-D pixel distance between eye corners 33 and 263",
            "field_names": [
                "left_iris_diameter_over_eye_span",
                "right_iris_diameter_over_eye_span",
            ],
        }
        return features, blink_detected, metrics, dict(self.last_iris_measurement_metadata)

    def _append_calibration_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_ids: Sequence[Hashable] | None,
        visit_ids: Sequence[Hashable] | None,
        frame_ids: Sequence[Hashable] | None,
    ) -> int:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        if X.shape[0] != y.shape[0] or y.shape[1] != 2:
            raise ValueError("calibration X/y must have matching rows and 2-D targets")
        if X.ndim != 2 or X.shape[1] != self.feature_schema["feature_count"]:
            raise ValueError("calibration features do not match the active formula schema")
        for entry in self._calibration_ledger:
            if entry.get("feature_formula_version", FEATURE_FORMULA_V1) != self.feature_formula_version:
                raise ValueError("calibration ledger mixes feature formulas")
        count = X.shape[0]
        if target_ids is None and visit_ids is None and frame_ids is None:
            visit = f"visit-{self._next_visit_number}"
            self._next_visit_number += 1
            target_ids = [tuple(float(value) for value in row) for row in y]
            visit_ids = [visit] * count
            frame_ids = [f"{visit}-frame-{index}" for index in range(count)]
        elif target_ids is None or visit_ids is None or frame_ids is None:
            raise ValueError("target_ids, visit_ids, and frame_ids must be supplied together")
        if not (len(target_ids) == len(visit_ids) == len(frame_ids) == count):
            raise ValueError("calibration group metadata must match the sample count")

        added = 0
        for features, target, target_id, visit_id, frame_id in zip(
            X, y, target_ids, visit_ids, frame_ids
        ):
            if not np.all(np.isfinite(features)) or not np.all(np.isfinite(target)):
                continue
            try:
                hash(target_id)
                hash(visit_id)
                hash(frame_id)
            except TypeError as exc:
                raise TypeError("target, visit, and frame IDs must be hashable") from exc
            key = (repr(visit_id), repr(frame_id))
            if key in self._seen_calibration_frames:
                continue
            self._seen_calibration_frames.add(key)
            self._calibration_ledger.append(
                {
                    "features": features.astype(np.float32).tolist(),
                    "feature_formula_version": self.feature_formula_version,
                    "target_px": target.astype(float).tolist(),
                    "target_id": target_id,
                    "visit_id": visit_id,
                    "frame_id": frame_id,
                }
            )
            added += 1
        return added

    @property
    def calibration_sample_count(self) -> int:
        return len(self._calibration_ledger)

    def _ledger_arrays(self):
        X = np.asarray([entry["features"] for entry in self._calibration_ledger], dtype=np.float32)
        y = np.asarray([entry["target_px"] for entry in self._calibration_ledger], dtype=np.float64)
        targets = [entry["target_id"] for entry in self._calibration_ledger]
        visits = [entry["visit_id"] for entry in self._calibration_ledger]
        return X, y, targets, visits

    @staticmethod
    def _balanced_sample_weights(target_ids: Sequence[Hashable], visit_ids: Sequence[Hashable]) -> np.ndarray:
        if len(target_ids) != len(visit_ids) or not target_ids:
            raise ValueError("target and visit groups must be nonempty and aligned")
        group_counts: dict[tuple[Hashable, Hashable], int] = {}
        target_visits: dict[Hashable, set[Hashable]] = {}
        for target, visit in zip(target_ids, visit_ids):
            group_counts[(target, visit)] = group_counts.get((target, visit), 0) + 1
            target_visits.setdefault(target, set()).add(visit)
        target_count = len(target_visits)
        weights = []
        for target, visit in zip(target_ids, visit_ids):
            visit_count = len(target_visits[target])
            rows_in_group = group_counts[(target, visit)]
            weights.append(1.0 / (target_count * visit_count * rows_in_group))
        weights = np.asarray(weights, dtype=np.float64)
        return weights * (len(weights) / np.sum(weights))

    def _record_training_metadata(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_ids: Sequence[Hashable] | None,
        visit_ids: Sequence[Hashable] | None,
        weights: np.ndarray | None,
        fit_residual: float | None,
        visit_holdout: dict[str, Any],
        balance_method: str,
        selected_indices: Sequence[int] | None,
        variable_scaling: np.ndarray | None,
    ) -> None:
        # The fitted matrix and metadata must describe the active formula.
        self._loaded_legacy_formula = False
        self._sync_feature_metadata()
        metadata = self.model.metadata
        scaler = getattr(self.model, "scaler", None)
        if scaler is not None:
            metadata["fitted_scaler"] = {
                "class": f"{type(scaler).__module__}.{type(scaler).__qualname__}",
                "with_mean": getattr(scaler, "with_mean", None),
                "with_std": getattr(scaler, "with_std", None),
                "n_features_in": int(getattr(scaler, "n_features_in_", X.shape[1])),
                "mean": np.asarray(getattr(scaler, "mean_", [])).tolist(),
                "scale": np.asarray(getattr(scaler, "scale_", [])).tolist(),
                "var": np.asarray(getattr(scaler, "var_", [])).tolist(),
            }
        native_model = getattr(self.model, "model", None)
        if native_model is None:
            native_model = getattr(self.model, "_template", None)
        get_params = getattr(native_model, "get_params", None)
        metadata["native_model"] = {
            "class": (
                f"{type(native_model).__module__}.{type(native_model).__qualname__}"
                if native_model is not None
                else f"{type(self.model).__module__}.{type(self.model).__qualname__}"
            ),
            "parameters": get_params(deep=False) if callable(get_params) else {},
        }
        training = metadata.setdefault("training", {})
        training.update(
            {
                "sample_count": int(len(X)),
                "target_visit_balance": (
                    "equal total target weight; equal visit weight within target"
                    if weights is not None
                    else "no_balance"
                ),
                "calibration_ledger_version": CALIBRATION_LEDGER_VERSION,
                "calibration_ledger": copy.deepcopy(self._calibration_ledger),
                "calibration_group_count": len(set(zip(target_ids or [], visit_ids or []))),
                "training_fit_residual_px": fit_residual,
                "residual_method_version": FIT_RESIDUAL_METHOD_VERSION,
                "visit_holdout": visit_holdout,
                "sample_weight_policy": "target_visit_balanced" if weights is not None else balance_method,
                "balance_method": balance_method,
                "selected_indices": list(selected_indices) if selected_indices is not None else list(range(len(X))),
                "variable_scaling": (
                    np.asarray(variable_scaling, dtype=np.float64).tolist()
                    if variable_scaling is not None
                    else None
                ),
                "raw_ledger_sample_count": len(self._calibration_ledger),
                "balance_definition": (
                    "equal total weight per target; visits split target weight evenly; "
                    "frames split visit weight evenly"
                    if weights is not None
                    else "no group balancing; explicitly selected unweighted fit"
                ),
                "head_descriptor_training_range": [
                    [float(np.min(X[:, index])), float(np.max(X[:, index]))]
                    for index in head_descriptor_indices(self.feature_formula_version)
                ],
                "head_descriptor_fields": [self.feature_schema["ordered_fields"][index] for index in head_descriptor_indices(self.feature_formula_version)],
                "target_pixel_range": [
                    [float(np.min(y[:, axis])), float(np.max(y[:, axis]))]
                    for axis in range(2)
                ],
            }
        )

    def _visit_holdout_residual(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_ids: Sequence[Hashable],
        visit_ids: Sequence[Hashable],
        variable_scaling: np.ndarray | None,
        use_group_weights: bool,
    ) -> dict[str, Any]:
        visits = list(dict.fromkeys(visit_ids))
        if len(visits) < 2:
            return {
                "method": "holdout_last_visit",
                "independence_status": "internal_not_independent",
                "value_px": None,
                "reason": "fewer_than_two_visits",
            }
        holdout_visit = visits[-1]
        holdout_mask = np.asarray([visit == holdout_visit for visit in visit_ids], dtype=bool)
        train_mask = ~holdout_mask
        if not np.any(train_mask) or not np.any(holdout_mask):
            return {
                "method": "holdout_last_visit",
                "independence_status": "internal_not_independent",
                "value_px": None,
                "reason": "empty_train_or_holdout_partition",
            }

        train_target_set = {target for target, keep in zip(target_ids, train_mask) if keep}
        holdout_target_set = {target for target, keep in zip(target_ids, holdout_mask) if keep}
        if not holdout_target_set.issubset(train_target_set):
            return {
                "method": "holdout_last_visit",
                "independence_status": "internal_not_independent",
                "value_px": None,
                "reason": "insufficient_repeated_target_visits",
            }

        model = copy.deepcopy(self.model)
        train_targets = [target for target, keep in zip(target_ids, train_mask) if keep]
        train_visits = [visit for visit, keep in zip(visit_ids, train_mask) if keep]
        train_X = X[train_mask]
        train_y = y[train_mask]
        sample_weights = None
        if use_group_weights and target_ids and visit_ids:
            if not getattr(model, "supports_sample_weight", False):
                return {
                    "method": "holdout_last_visit",
                    "independence_status": "internal_not_independent",
                    "value_px": None,
                    "reason": "model_does_not_support_sample_weight",
                }
            sample_weights = self._balanced_sample_weights(
                train_targets,
                train_visits,
            )

        if getattr(model, "supports_sample_weight", False) and sample_weights is not None:
            model.train(
                train_X,
                train_y,
                variable_scaling=variable_scaling,
                sample_weight=sample_weights,
            )
        else:
            model.train(train_X, train_y, variable_scaling=variable_scaling)
        predictions = model.predict(X[holdout_mask])
        residuals = np.linalg.norm(predictions - y[holdout_mask], axis=1)
        heldout_targets = [target for target, keep in zip(target_ids, holdout_mask) if keep]
        residuals_by_target: dict[Hashable, list[float]] = {}
        for target, residual in zip(heldout_targets, residuals):
            residuals_by_target.setdefault(target, []).append(float(residual))
        target_balanced_residual = float(
            np.mean([np.mean(values) for values in residuals_by_target.values()])
        )
        return {
            "method": "holdout_last_visit",
            "independence_status": "internal_not_independent",
            "visit_id": holdout_visit,
            "metric": "target_balanced_mean_euclidean_visit_holdout_residual_px",
            "value_px": target_balanced_residual,
            "target_count": len(residuals_by_target),
            "sample_count": int(np.count_nonzero(holdout_mask)),
        }

    def train(
        self,
        X,
        y,
        variable_scaling=None,
        *,
        target_ids: Sequence[Hashable] | None = None,
        visit_ids: Sequence[Hashable] | None = None,
        frame_ids: Sequence[Hashable] | None = None,
        use_group_weights: bool | None = None,
        accumulate: bool = True,
        feature_formula_version: str | None = None,
    ):
        """Fit only the selected formula; tagged calls refit retained visits."""
        if self.feature_formula_version == LEGACY_UNVERSIONED_FORMULA:
            raise ValueError("legacy-unversioned models are prediction-only; create a fresh estimator for recalibration")
        if feature_formula_version is not None and feature_formula_version != self.feature_formula_version:
            raise ValueError("training feature formula does not match the active extractor")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.feature_schema["feature_count"]:
            raise ValueError("training features do not match the active formula schema")
        y = np.asarray(y)
        tagged = any(value is not None for value in (target_ids, visit_ids, frame_ids))
        should_ledger = tagged or (accumulate and bool(self._calibration_ledger))
        if use_group_weights is None:
            use_group_weights = bool(self.calibration_use_group_weights)
        if should_ledger:
            self._append_calibration_samples(X, y, target_ids, visit_ids, frame_ids)
            X_all, y_all, fit_targets, fit_visits = self._ledger_arrays()
            weights = (
                self._balanced_sample_weights(fit_targets, fit_visits)
                if use_group_weights and fit_targets
                else None
            )
            supports_weights = bool(getattr(self.model, "supports_sample_weight", False))
            selected_indices = np.arange(len(X_all), dtype=np.int64)
            balance_method = (
                "target_visit_balanced_sample_weight" if weights is not None else "unweighted_legacy_explicit"
            )
            if weights is not None and not supports_weights:
                raise NotImplementedError(
                    f"{self.model_name} does not support group-weighted calibration. "
                    "Choose Ridge for target/visit-balanced weighted fitting, or explicitly set "
                    "estimator.calibration_use_group_weights=False for the legacy unweighted fit. "
                    "Raw calibration samples remain in the ledger."
                )
            else:
                X_fit, y_fit = X_all, y_all
            if weights is not None and supports_weights:
                self.model.train(
                    X_fit,
                    y_fit,
                    variable_scaling=variable_scaling,
                    sample_weight=weights,
                )
            else:
                self.model.train(X_fit, y_fit, variable_scaling=variable_scaling)
            predictions = self.model.predict(X_fit)
            residuals = np.linalg.norm(predictions - y_fit, axis=1)
            fit_residual = float(np.mean(residuals))
            visit_holdout = self._visit_holdout_residual(
                X_all,
                y_all,
                self._ledger_arrays()[2],
                self._ledger_arrays()[3],
                variable_scaling,
                use_group_weights,
            )
            self._record_training_metadata(
                X_fit,
                y_fit,
                fit_targets,
                fit_visits,
                weights,
                fit_residual,
                visit_holdout,
                balance_method,
                selected_indices,
                variable_scaling,
            )
            return

        X_fit, y_fit = np.asarray(X), np.asarray(y)
        self.model.train(X_fit, y_fit, variable_scaling=variable_scaling)
        predictions = self.model.predict(X_fit)
        residuals = np.linalg.norm(predictions - y_fit, axis=1)
        self._record_training_metadata(
            X_fit,
            y_fit,
            None,
            None,
            None,
            float(np.mean(residuals)),
            {
                "method": "holdout_last_visit",
                "independence_status": "internal_not_independent",
                "value_px": None,
                "reason": "no_visit_groups",
            },
            "unweighted_legacy",
            None,
            variable_scaling,
        )

    def predict(self, X, *, feature_formula_version: str | None = None):
        metadata = getattr(self.model, "metadata", {})
        formula_version = metadata.get("feature_formula_version") if isinstance(metadata, dict) else None
        legacy = bool(
            isinstance(metadata, dict)
            and (
                metadata.get("legacy")
                or metadata.get("feature_contract_status") == "legacy_unversioned"
                or formula_version in (None, LEGACY_UNVERSIONED_FORMULA)
            )
        )
        if feature_formula_version is not None and feature_formula_version != self.feature_formula_version:
            raise ValueError("prediction feature formula does not match the active extractor")
        if formula_version and formula_version not in (
            FEATURE_FORMULA_V1, FEATURE_FORMULA_V2, LEGACY_UNVERSIONED_FORMULA,
        ):
            raise ValueError(
                f"model expects feature formula {formula_version!r}; current extractor is "
                f"{FEATURE_FORMULA_VERSION!r}. Same-length feature vectors cannot be mixed silently."
            )
        if formula_version in {FEATURE_FORMULA_V1, FEATURE_FORMULA_V2}:
            _validate_versioned_feature_contract(self.model, self.feature_formula_version)
        elif self.feature_formula_version != LEGACY_UNVERSIONED_FORMULA:
            raise ValueError("unversioned model must be explicitly loaded on its historical path")
        if legacy and not self._loaded_legacy_formula:
            warnings.warn(
                "Loaded model has no recorded feature formula; using the explicit legacy-unversioned "
                "prediction path for compatibility.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._loaded_legacy_formula = True

        X = np.asarray(X)
        single = X.ndim == 1
        X_rows = X.reshape(1, -1) if single else X
        expected_feature_count = self.feature_schema["feature_count"]
        if X_rows.ndim != 2 or X_rows.shape[1] != expected_feature_count:
            raise ValueError(
                f"feature shape {X_rows.shape} does not match current schema width "
                f"{expected_feature_count}"
            )
        training = metadata.get("training", {}) if isinstance(metadata, dict) else {}
        ranges = training.get("head_descriptor_training_range") if isinstance(training, dict) else None
        flags: list[bool | None] = []
        indices = head_descriptor_indices(self.feature_formula_version)
        if ranges and len(ranges) == len(indices):
            for row in X_rows:
                descriptors = row[list(indices)]
                flags.append(
                    any(
                        value < bounds[0] or value > bounds[1]
                        for value, bounds in zip(descriptors, ranges)
                    )
                )
        else:
            flags = [None] * len(X_rows)
        self.last_runtime_flags = {
            "head_descriptor_outside_training_range": flags[0] if single else flags,
            "feature_formula_legacy_unversioned": legacy,
        }
        return self.model.predict(X)

    def save_model(self, path: str | Path):
        if self.feature_formula_version in {FEATURE_FORMULA_V1, FEATURE_FORMULA_V2}:
            self._sync_feature_metadata()
            _validate_versioned_feature_contract(self.model, self.feature_formula_version)
        return self.model.save(path)

    def load_model(self, path: str | Path):
        model = BaseModel.load(path)
        metadata = getattr(model, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            model.metadata = metadata
        formula_version = metadata.get("feature_formula_version")
        if formula_version and formula_version not in {FEATURE_FORMULA_V1, FEATURE_FORMULA_V2, LEGACY_UNVERSIONED_FORMULA}:
            raise ValueError(
                f"model feature formula {formula_version!r} is incompatible with current "
                f"extractor {FEATURE_FORMULA_VERSION!r}"
            )
        if formula_version in {FEATURE_FORMULA_V1, FEATURE_FORMULA_V2}:
            _validate_versioned_feature_contract(model)
        if not formula_version or formula_version == LEGACY_UNVERSIONED_FORMULA:
            self._loaded_legacy_formula = True
            self.feature_formula_version = LEGACY_UNVERSIONED_FORMULA
            metadata.setdefault("feature_formula_path", "legacy-unversioned-pickle")
            metadata.setdefault("feature_contract_status", "legacy_unversioned")
            warnings.warn(
                "Loaded a pre-metadata model; retaining its legacy-unversioned feature path. "
                "The historical extractor schema is assumed for prediction.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            self.feature_formula_version = formula_version
            self._loaded_legacy_formula = False
        self.feature_schema = _feature_schema(self.feature_formula_version)
        self.screen_info = copy.deepcopy(metadata.get("calibration_canvas_size_px", metadata.get("screen_info_px")))
        self._model_loaded = True
        self.model = model
        training = metadata.get("training", {})
        ledger = training.get("calibration_ledger", []) if isinstance(training, dict) else []
        if isinstance(ledger, list):
            self._calibration_ledger = copy.deepcopy(ledger)
            self._seen_calibration_frames = {
                (repr(item.get("visit_id")), repr(item.get("frame_id")))
                for item in ledger
                if isinstance(item, dict)
            }
        return self.model


__all__ = [
    "CALIBRATION_LEDGER_VERSION",
    "FEATURE_FORMULA_VERSION",
    "FIT_RESIDUAL_METHOD_VERSION",
    "LEGACY_IRIS_FORMULA_VERSION",
    "LEGACY_UNVERSIONED_FORMULA",
    "EyeStateTracker",
    "GazeEstimator",
    "compute_pixel_ear",
]
