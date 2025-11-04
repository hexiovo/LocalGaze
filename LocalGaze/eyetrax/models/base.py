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

    def __init__(self) -> None:
        self.scaler = StandardScaler()

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
    ) -> None:
        self.variable_scaling = variable_scaling
        Xs = self.scaler.fit_transform(X)
        if variable_scaling is not None:
            Xs *= variable_scaling
        self._native_train(Xs, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        if getattr(self, "variable_scaling", None) is not None:
            Xs *= self.variable_scaling
        return self._native_predict(Xs)

    def save(self, relative_path: str) -> None:
        """
        将模型保存到指定相对路径。

        Args:
            relative_path: 相对于当前工作目录的路径，例如 "models/gaze_model.pkl"
        """
        # 确保父目录存在
        full_path = Path(relative_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存模型
        with full_path.open("wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, relative_path: str) -> "BaseModel":
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

        return model
