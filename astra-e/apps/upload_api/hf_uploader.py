"""Hugging Face Hub integration for persisting raw experimental video dataset."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .schemas import RecordingMetadata

logger = logging.getLogger("astra.collector.hf")


class HuggingFaceDatasetUploader:
    """Manages secure server-side upload of raw experimental videos to Hugging Face Hub."""

    def __init__(
        self,
        repo_id: str | None = None,
        token: str | None = None,
        mock_mode: bool = False,
    ) -> None:
        self.repo_id = repo_id or os.environ.get("HF_RAW_DATASET_REPO", "na124441/astra-e-raw")
        self.token = token or os.environ.get("HF_TOKEN")
        self.mock_mode = (
            mock_mode
            or os.environ.get("MOCK_HF_UPLOAD", "").lower() in ("true", "1", "yes")
            or not self.token
        )

        if not self.mock_mode:
            try:
                from huggingface_hub import HfApi
                self.api: Any = HfApi(token=self.token)
                logger.info("Initialized Hugging Face HfApi client for repo: %s", self.repo_id)
            except Exception as e:
                logger.warning(
                    "Failed to initialize Hugging Face Hub API (%s). Falling back to mock mode.", e
                )
                self.mock_mode = True
                self.api = None
        else:
            self.api = None
            logger.info("Hugging Face Hub uploader running in MOCK/OFFLINE mode for %s.", self.repo_id)

    def upload_recording(
        self,
        video_path: Path | str,
        metadata: RecordingMetadata,
    ) -> tuple[str, str]:
        """
        Uploads video and accompanying metadata to the Hugging Face dataset repo.
        Remote path convention:
          videos/exp001/{run_id}/{camera_id}.mp4
          videos/exp001/{run_id}/{camera_id}.json

        Returns tuple of (remote_video_path, remote_metadata_path).
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found at {video_path}")

        exp_id = metadata.experiment_id.lower()
        run_id = metadata.run_id
        cam_id = metadata.camera_id

        remote_video_path = f"videos/{exp_id}/{run_id}/{cam_id}.mp4"
        remote_metadata_path = f"videos/{exp_id}/{run_id}/{cam_id}.json"

        metadata_dict = metadata.model_dump()
        metadata_bytes = json.dumps(metadata_dict, indent=2).encode("utf-8")

        if self.mock_mode:
            logger.info(
                "[MOCK HF] Successfully uploaded %s -> %s and metadata -> %s (mock mode)",
                video_path.name,
                remote_video_path,
                remote_metadata_path,
            )
            return remote_video_path, remote_metadata_path

        try:
            # 1. Upload video file
            self.api.upload_file(
                path_or_fileobj=str(video_path),
                path_in_repo=remote_video_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"Add video {exp_id}/{run_id}/{cam_id} by {metadata.collector_id}",
            )

            # 2. Upload metadata JSON
            self.api.upload_file(
                path_or_fileobj=metadata_bytes,
                path_in_repo=remote_metadata_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"Add metadata {exp_id}/{run_id}/{cam_id} by {metadata.collector_id}",
            )

            logger.info(
                "Uploaded %s and metadata to HF repo %s",
                remote_video_path,
                self.repo_id,
            )
            return remote_video_path, remote_metadata_path

        except Exception as e:
            logger.error("Hugging Face Hub upload failed for %s: %s", remote_video_path, e)
            raise RuntimeError(f"Hugging Face upload error: {e}") from e
