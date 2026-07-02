from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import cv2

from .config import Settings
from .pipeline import EmotionPipeline


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a facial emotion recognition pipeline")
    parser.add_argument("--image", type=str, help="Path to an image file to analyze")
    parser.add_argument("--camera", action="store_true", help="Run the webcam pipeline")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index")
    parser.add_argument("--output", type=str, help="Optional output path for analyzed images")
    parser.add_argument("--no-display", action="store_true", help="Disable the preview window")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    settings = Settings.load()
    pipeline = EmotionPipeline(settings)

    if args.image:
        image_path = Path(args.image)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        result = pipeline.process_frame(image)
        annotated = result["frame"]
        print(json.dumps(result["results"], indent=2))

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated)
            logging.info("Saved output image to %s", output_path)
        return 0

    if args.camera or not args.image:
        return run_camera(pipeline, camera_index=args.camera_index, show_window=not args.no_display)

    return 0


def run_camera(pipeline: EmotionPipeline, camera_index: int = 0, show_window: bool = True) -> int:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")

    logging.info("Starting webcam stream. Press 'q' to quit.")
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            logging.error("Could not read frame from camera")
            break

        result = pipeline.process_frame(frame)
        if show_window:
            cv2.imshow("Facial Emotion Recognition", result["frame"])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    if show_window:
        cv2.destroyAllWindows()
    return 0
