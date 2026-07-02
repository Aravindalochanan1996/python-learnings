from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EmotionResult:
    label: str
    confidence: float
    details: dict[str, Any]


class EmotionClassifier:
    """Classifies a detected face into a simple emotion label."""

    def classify(self, features: dict[str, Any]) -> EmotionResult:
        if not features.get("face_detected", False):
            return EmotionResult(
                label="unknown",
                confidence=0.0,
                details={"reason": "No face detected"},
            )

        smile_detected = bool(features.get("smile_detected", False))
        eye_count = int(features.get("eye_count", 0))

        if smile_detected:
            return EmotionResult(
                label="happy",
                confidence=0.88 if eye_count >= 1 else 0.76,
                details={"smile_detected": True, "eye_count": eye_count},
            )

        if eye_count >= 2:
            return EmotionResult(
                label="neutral",
                confidence=0.72,
                details={"smile_detected": False, "eye_count": eye_count},
            )

        return EmotionResult(
            label="neutral",
            confidence=0.6,
            details={"smile_detected": False, "eye_count": eye_count},
        )
