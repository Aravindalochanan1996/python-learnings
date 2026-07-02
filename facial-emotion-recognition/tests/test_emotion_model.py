from app.emotion_model import EmotionClassifier


def test_classify_happy_when_smile_detected() -> None:
    classifier = EmotionClassifier()

    result = classifier.classify(
        {"face_detected": True, "smile_detected": True, "eye_count": 2}
    )

    assert result.label == "happy"
    assert result.confidence >= 0.75


def test_classify_neutral_when_no_smile_is_detected() -> None:
    classifier = EmotionClassifier()

    result = classifier.classify(
        {"face_detected": True, "smile_detected": False, "eye_count": 0}
    )

    assert result.label == "neutral"
    assert result.confidence >= 0.5
