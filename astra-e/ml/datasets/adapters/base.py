"""Base abstract adapter for converting raw external datasets into standardized ASTRA-E contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ml.datasets.schemas import ActionSegmentAnnotation, RecordingMetadata


class BaseDatasetAdapter(ABC):
    """
    Abstract Base Class for external dataset ingestion and standardization.
    Translates third-party video and annotation schemas into ASTRA-E RecordingMetadata contracts.
    """

    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root)

    @abstractmethod
    def load_dataset(self) -> list[RecordingMetadata]:
        """Parse raw dataset manifests and produce structured RecordingMetadata records."""
        raise NotImplementedError

    @abstractmethod
    def get_recording_video_path(self, recording_id: str) -> Path:
        """Resolve absolute file path to the raw video recording."""
        raise NotImplementedError

    @abstractmethod
    def get_annotations(self, recording_id: str) -> list[ActionSegmentAnnotation]:
        """Return standardized temporal action segment annotations for a given recording."""
        raise NotImplementedError

    def export_astra_manifest(self, output_file: str | Path) -> Path:
        """Serialize standardized recording metadata to a unified JSON catalog."""
        import json

        recordings = self.load_dataset()
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        catalog = {
            "adapter": self.__class__.__name__,
            "dataset_root": str(self.dataset_root),
            "recordings_count": len(recordings),
            "recordings": [r.model_dump() for r in recordings],
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)

        return out_path
