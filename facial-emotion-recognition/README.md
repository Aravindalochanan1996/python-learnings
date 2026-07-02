# Facial Emotion Recognition

A production-ready starter application for detecting faces and classifying simple emotion states from images or a webcam feed.

## Features

- Modular Python package structure
- CLI entry point for image or webcam analysis
- OpenCV-based face and facial-feature detection
- Basic emotion classification pipeline
- Unit tests for the classification logic

## Project structure

- `main.py` - application entry point
- `app/` - core application package
  - `config.py` - environment-based configuration
  - `detectors.py` - face and feature detection
  - `emotion_model.py` - emotion classification logic
  - `pipeline.py` - orchestrates the analysis flow
  - `cli.py` - command-line interface
- `tests/` - automated tests
- `requirements.txt` - Python dependencies

## Setup

Activate the existing virtual environment from the `day4` folder:

```powershell
cd python-learnings/day4
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r facial-emotion-recognition/requirements.txt
```

## Run the app

Analyze an image:

```powershell
python .\facial-emotion-recognition\main.py --image .\facial-emotion-recognition\sample.jpg
```

Run the webcam stream:

```powershell
python .\facial-emotion-recognition\main.py --camera
```

## Run tests

```powershell
pytest .\facial-emotion-recognition\tests
```

## Notes

This starter is designed to be extended with a more advanced model later, but it already provides a clean, modular foundation for facial emotion analysis.
