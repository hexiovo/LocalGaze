from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from eyetrax.constants import LEFT_EYE_INDICES, MUTUAL_INDICES, RIGHT_EYE_INDICES
from eyetrax.models import BaseModel, create_model


class GazeEstimator:
    def __init__(
        self,
        model_name: str = "ridge",
        model_kwargs: dict | None = None,
        ear_history_len: int = 50,
        blink_threshold_ratio: float = 0.8,
        min_history: int = 15,
    ):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        self.model: BaseModel = create_model(model_name, **(model_kwargs or {}))

        self._ear_history = deque(maxlen=ear_history_len)
        self._blink_ratio = blink_threshold_ratio
        self._min_history = min_history

    def extract_features(self, image ,return_more=False):
        """
        Takes in image and returns landmarks around the eye region
        Normalization with nose tip as anchor
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            if return_more:
                return None, False, None, None
            else:
                return None, False

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        all_points = np.array(
            [(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32
        )
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
        R = np.column_stack((x_axis, y_axis, z_axis))
        rotated_points = (R.T @ shifted_points.T).T

        left_corner_rot = R.T @ (left_corner - nose_anchor)
        right_corner_rot = R.T @ (right_corner - nose_anchor)
        inter_eye_dist = np.linalg.norm(right_corner_rot - left_corner_rot)
        if inter_eye_dist > 1e-7:
            rotated_points /= inter_eye_dist

        subset_indices = LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MUTUAL_INDICES
        eye_landmarks = rotated_points[subset_indices]
        features = eye_landmarks.flatten()

        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        roll = np.arctan2(R[2, 1], R[2, 2])
        features = np.concatenate([features, [yaw, pitch, roll]])

        # Blink detection
        left_eye_inner = np.array([landmarks[133].x, landmarks[133].y])
        left_eye_outer = np.array([landmarks[33].x, landmarks[33].y])
        left_eye_top = np.array([landmarks[159].x, landmarks[159].y])
        left_eye_bottom = np.array([landmarks[145].x, landmarks[145].y])

        right_eye_inner = np.array([landmarks[362].x, landmarks[362].y])
        right_eye_outer = np.array([landmarks[263].x, landmarks[263].y])
        right_eye_top = np.array([landmarks[386].x, landmarks[386].y])
        right_eye_bottom = np.array([landmarks[374].x, landmarks[374].y])

        left_eye_width = np.linalg.norm(left_eye_outer - left_eye_inner)
        left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
        left_EAR = left_eye_height / (left_eye_width + 1e-9)

        right_eye_width = np.linalg.norm(right_eye_outer - right_eye_inner)
        right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
        right_EAR = right_eye_height / (right_eye_width + 1e-9)

        EAR = (left_EAR + right_EAR) / 2

        self._ear_history.append(EAR)
        if len(self._ear_history) >= self._min_history:
            thr = float(np.mean(self._ear_history)) * self._blink_ratio
        else:
            thr = 0.2
        blink_detected = EAR < thr

        LEFT_IRIS_INDICES = [473, 474, 475, 476, 477]  # refine=True 模式下
        RIGHT_IRIS_INDICES = [468, 469, 470, 471, 472]


        left_iris_points = np.array([(landmarks[i].x, landmarks[i].y) for i in LEFT_IRIS_INDICES])
        right_iris_points = np.array([(landmarks[i].x, landmarks[i].y) for i in RIGHT_IRIS_INDICES])

        # 分离中心点
        left_center = left_iris_points[-1]  # 477 是中心
        right_center = right_iris_points[-1]  # 472 是中心

        # 计算平均半径（中心到其他边缘点的平均距离）
        left_radius = np.mean([np.linalg.norm(p - left_center) for p in left_iris_points[:-1]])
        right_radius = np.mean([np.linalg.norm(p - right_center) for p in right_iris_points[:-1]])

        # 归一化后得到“直径”
        left_pupil_diameter = (2 * left_radius) / inter_eye_dist
        right_pupil_diameter = (2 * right_radius) / inter_eye_dist


        if return_more:
            return features, blink_detected, left_pupil_diameter, right_pupil_diameter
        else:
            return features, blink_detected


    def save_model(self, path: str | Path):
        """
        Pickle model
        """
        self.model.save(path)

    def load_model(self, path: str | Path):
        self.model = BaseModel.load(path)

    def train(self, X, y, variable_scaling=None):
        """
        Trains gaze prediction model
        """
        self.model.train(X, y, variable_scaling)

    def predict(self, X):
        """
        Predicts gaze location
        """
        return self.model.predict(X)
