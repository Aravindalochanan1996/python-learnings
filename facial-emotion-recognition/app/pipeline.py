from __future__ import annotations

from typing import Any

import cv2

from .config import Settings
from .detectors import FaceDetector
from .emotion_model import EmotionClassifier, EmotionResult


class EmotionPipeline:
    """Coordinates face detection and emotion classification."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.detector = FaceDetector(settings)
        self.classifier = EmotionClassifier()

    def process_frame(self, frame: Any) -> dict[str, Any]:
        detected_faces = self.detector.detect_faces(frame)
        results: list[dict[str, Any]] = []

        for face in detected_faces:
            analysis = self.detector.analyze_face(frame, face)
            features = {
                "face_detected": True,
                "smile_detected": analysis["smile_detected"],
                "eye_count": analysis["eye_count"],
            }
            emotion: EmotionResult = self.classifier.classify(features)
            results.append(
                {
                    "face_box": analysis["face_box"],
                    "emotion": {
                        "label": emotion.label,
                        "confidence": emotion.confidence,
                        "details": emotion.details,
                    },
                }
            )

        annotated = self._annotate_frame(frame, results)
        return {
            "frame": annotated,
            "results": results,
            "face_count": len(results),
        }

    def _annotate_frame(self, frame: Any, results: list[dict[str, Any]]) -> Any:
        annotated = frame.copy()
        for item in results:
            x, y, w, h = item["face_box"]
            emotion_label = item["emotion"]["label"]
            confidence = item["emotion"]["confidence"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{emotion_label} ({confidence:.2f})"
            cv2.putText(
                annotated,
                label,
                (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        return annotated
