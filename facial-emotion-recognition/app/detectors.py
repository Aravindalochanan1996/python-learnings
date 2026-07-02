from __future__ import annotations

from typing import Any

import cv2

from .config import Settings


class FaceDetector:
    """Detects faces and simple facial features with OpenCV cascades."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

    def detect_faces(self, image: Any) -> list[tuple[int, int, int, int]]:
        if image is None:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.settings.scale_factor,
            minNeighbors=self.settings.min_neighbors,
            minSize=(self.settings.min_face_size, self.settings.min_face_size),
        )
        return [tuple(map(int, face)) for face in faces]

    def analyze_face(self, image: Any, face: tuple[int, int, int, int]) -> dict[str, Any]:
        x, y, w, h = face
        roi = image[y : y + h, x : x + w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        smiles = self.smile_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=1.7,
            minNeighbors=20,
            minSize=(25, 25),
        )
        eyes = self.eye_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(15, 15),
        )

        return {
            "face_box": face,
            "smile_detected": len(smiles) > 0,
            "eye_count": len(eyes),
        }
