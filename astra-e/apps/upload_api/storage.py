"""Server-side chunk staging, file assembly, and cryptographic checksum validation."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import BinaryIO


class ChunkStorageManager:
    """Manages temporary storage of uploaded video chunks and file reassembly."""

    def __init__(self, base_staging_dir: Path | str | None = None) -> None:
        if base_staging_dir is None:
            self.base_staging_dir = Path(__file__).resolve().parent.parent.parent / "data" / "collector_staging"
        else:
            self.base_staging_dir = Path(base_staging_dir)
        self.base_staging_dir.mkdir(parents=True, exist_ok=True)

    def get_upload_dir(self, upload_id: str) -> Path:
        """Return and ensure staging directory for a specific upload."""
        upload_dir = self.base_staging_dir / upload_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        return upload_dir

    def write_chunk(self, upload_id: str, chunk_index: int, chunk_data: bytes) -> tuple[int, str]:
        """Write chunk payload to disk and return (bytes_written, chunk_sha256)."""
        upload_dir = self.get_upload_dir(upload_id)
        chunk_path = upload_dir / f"chunk_{chunk_index:05d}.part"

        sha256_hasher = hashlib.sha256()
        sha256_hasher.update(chunk_data)
        chunk_sha256 = sha256_hasher.hexdigest()

        with open(chunk_path, "wb") as f:
            f.write(chunk_data)

        return len(chunk_data), chunk_sha256

    def list_uploaded_chunks(self, upload_id: str) -> set[int]:
        """Return set of chunk indices successfully saved."""
        upload_dir = self.base_staging_dir / upload_id
        if not upload_dir.exists():
            return set()
        chunks = set()
        for p in upload_dir.glob("chunk_*.part"):
            try:
                idx = int(p.stem.split("_")[1])
                chunks.add(idx)
            except (IndexError, ValueError):
                continue
        return chunks

    def assemble_file(self, upload_id: str, total_chunks: int, output_path: Path) -> str:
        """
        Assemble chunks in sequential order into output_path and compute full SHA-256.
        Raises ValueError if any chunk is missing.
        """
        upload_dir = self.get_upload_dir(upload_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sha256_hasher = hashlib.sha256()

        with open(output_path, "wb") as out_f:
            for idx in range(total_chunks):
                chunk_path = upload_dir / f"chunk_{idx:05d}.part"
                if not chunk_path.exists():
                    raise FileNotFoundError(f"Missing chunk {idx} for upload {upload_id}")
                
                with open(chunk_path, "rb") as chunk_f:
                    while True:
                        buf = chunk_f.read(65536)
                        if not buf:
                            break
                        sha256_hasher.update(buf)
                        out_f.write(buf)

        return sha256_hasher.hexdigest()

    @staticmethod
    def compute_sha256(file_path: Path | str) -> str:
        """Compute SHA-256 of any file via streaming buffer."""
        sha256_hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                buf = f.read(65536)
                if not buf:
                    break
                sha256_hasher.update(buf)
        return sha256_hasher.hexdigest()

    def cleanup_staging(self, upload_id: str) -> None:
        """Remove staging chunks for an upload session."""
        upload_dir = self.base_staging_dir / upload_id
        if upload_dir.exists():
            shutil.rmtree(upload_dir, ignore_errors=True)
