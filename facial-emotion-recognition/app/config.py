from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Application configuration loaded from environment values."""

    project_root: Path
    camera_index: int = 0
    min_face_size: int = 30
    scale_factor: float = 1.1
    min_neighbors: int = 5
    enable_display: bool = True

    @classmethod
    def load(cls) -> "Settings":
        project_root = Path(__file__).resolve().parent.parent
        return cls(
            project_root=project_root,
            camera_index=int(os.getenv("CAMERA_INDEX", "0")),
            min_face_size=int(os.getenv("MIN_FACE_SIZE", "30")),
            scale_factor=float(os.getenv("SCALE_FACTOR", "1.1")),
            min_neighbors=int(os.getenv("MIN_NEIGHBORS", "5")),
            enable_display=os.getenv("ENABLE_DISPLAY", "true").lower() == "true",
        )
