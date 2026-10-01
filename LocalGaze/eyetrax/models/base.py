from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler


class BaseModel(ABC):
    """
    Common interface every gaze-prediction model must implement
    """

    supports_sample_weight = False

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        # GazeEstimator supplies the actual feature contract, never infer it
        # from the feature count or the pickle's creation date.
        self.metadata: dict = {
            "model_metadata_version": 1,
            "feature_contract_status": "unversioned",
        }

    @abstractmethod
    def _init_native(self, **kwargs): ...
    @abstractmethod
    def _native_train(self, X: np.ndarray, y: np.ndarray): ...
    @abstractmethod
    def _native_predict(self, X: np.ndarray) -> np.ndarray: ...

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_scaling: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        X = self._feature_matrix(X)
        y = np.asarray(y, dtype=float)
        if y.ndim not in (1, 2) or y.shape[0] != X.shape[0]:
            raise ValueError("y must have one target row per feature row")
        if y.ndim == 2 and y.shape[1] == 0:
            raise ValueError("y must contain at least one target column")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values")
        scaling = self._scaling_vector(variable_scaling, X.shape[1])
        weights = None
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float)
            if weights.ndim != 1 or len(weights) != len(X):
                raise ValueError("sample_weight must have one value per feature row")
            if not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError("sample_weight must contain finite nonnegative values")
            total = float(np.sum(weights))
            if not np.isfinite(total) or total <= 0:
                raise ValueError("sample_weight must have a finite positive total")
            if not self.supports_sample_weight:
                raise NotImplementedError(
                    f"{type(self).__name__} does not support sample_weight; "
                    "weights cannot be silently discarded"
                )
            # Both stages ignore exactly the same zero-weight rows.
            positive = weights > 0
            X, y, weights = X[positive], y[positive], weights[positive]
        scaler = StandardScaler()
        if weights is None:
            Xs = scaler.fit_transform(X)
        else:
            scaler.fit(X, sample_weight=weights)
            Xs = scaler.transform(X)
        if scaling is not None:
            Xs *= scaling
        if weights is None:
            # Preserve the existing subclass API for nonweighted models.
            self._native_train(Xs, y)
        else:
            self._native_train(Xs, y, sample_weight=weights)
        self.scaler = scaler
        self.variable_scaling = scaling

    @staticmethod
    def _feature_matrix(X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or min(values.shape) == 0:
            raise ValueError("X must be a nonempty two-dimensional feature matrix")
        if not np.all(np.isfinite(values)):
            raise ValueError("X must contain only finite values")
        return values

    @staticmethod
    def _scaling_vector(values: np.ndarray | None, feature_count: int):
        if values is None:
            return None
        scaling = np.asarray(values, dtype=float)
        if scaling.shape != (feature_count,) or not np.all(np.isfinite(scaling)):
            raise ValueError("variable_scaling must be one finite value per feature")
        return scaling.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._feature_matrix(X)
        feature_count = getattr(self.scaler, "n_features_in_", None)
        if feature_count is None:
            raise ValueError("The model must be trained before prediction")
        if X.shape[1] != feature_count:
            raise ValueError(
                f"Expected {feature_count} features, received {X.shape[1]}"
            )
        Xs = self.scaler.transform(X)
        if getattr(self, "variable_scaling", None) is not None:
            Xs *= self.variable_scaling
        return self._native_predict(Xs)

    def save(self, relative_path: str | Path, *, overwrite: bool = False) -> None:
        """
        将模型保存到指定相对路径。

        Args:
            relative_path: 相对于当前工作目录的路径，例如 "models/gaze_model.pkl"
        """
        # 确保父目录存在
        full_path = Path(relative_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)

        if not isinstance(getattr(self, "metadata", None), dict):
            raise ValueError("Model metadata must be a dictionary")
        # Exclusive creation protects historical model files by default.
        with full_path.open("wb" if overwrite else "xb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, relative_path: str | Path) -> "BaseModel":
        """
        从指定相对路径加载模型。

        Args:
            relative_path: 相对于当前工作目录的路径，例如 "models/gaze_model.pkl"

        Returns:
            加载的 BaseModel 实例
        """
        full_path = Path(relative_path)

        if not full_path.is_file():
            raise FileNotFoundError(f"No model found at: {full_path.resolve()}")

        with full_path.open("rb") as fh:
            model = pickle.load(fh)
        if not isinstance(model, BaseModel):
            raise ValueError("The pickle does not contain an eyetrax BaseModel")
        if not hasattr(model, "metadata"):
            model.metadata = {
                "model_metadata_version": 0,
                "feature_contract_status": "legacy_unversioned",
                "legacy": True,
            }
        elif not isinstance(model.metadata, dict):
            raise ValueError("Model metadata must be a dictionary")
        return model
