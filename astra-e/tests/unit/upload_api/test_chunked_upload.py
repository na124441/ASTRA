"""Unit tests for chunked file upload, reassembly, and storage manager."""

import hashlib
from pathlib import Path
import pytest
from apps.upload_api.storage import ChunkStorageManager


def test_chunk_storage_manager_lifecycle(tmp_path):
    storage = ChunkStorageManager(base_staging_dir=tmp_path / "staging")
    upload_id = "test_upload_123"

    # 1. Create simulated multi-chunk payload (3 chunks of 1000 bytes)
    payload_chunk0 = b"A" * 1000
    payload_chunk1 = b"B" * 1000
    payload_chunk2 = b"C" * 1000
    expected_full_bytes = payload_chunk0 + payload_chunk1 + payload_chunk2
    expected_full_sha256 = hashlib.sha256(expected_full_bytes).hexdigest()

    # 2. Write chunks
    bytes0, hash0 = storage.write_chunk(upload_id, 0, payload_chunk0)
    assert bytes0 == 1000
    assert hash0 == hashlib.sha256(payload_chunk0).hexdigest()

    bytes1, hash1 = storage.write_chunk(upload_id, 1, payload_chunk1)
    assert bytes1 == 1000

    bytes2, hash2 = storage.write_chunk(upload_id, 2, payload_chunk2)
    assert bytes2 == 1000

    # 3. Check listed chunks
    uploaded_indices = storage.list_uploaded_chunks(upload_id)
    assert uploaded_indices == {0, 1, 2}

    # 4. Assemble
    assembled_file = tmp_path / "output.mp4"
    computed_sha256 = storage.assemble_file(upload_id, 3, assembled_file)
    assert computed_sha256 == expected_full_sha256
    assert assembled_file.exists()
    assert assembled_file.read_bytes() == expected_full_bytes

    # 5. Clean up
    storage.cleanup_staging(upload_id)
    assert len(storage.list_uploaded_chunks(upload_id)) == 0


def test_assemble_fails_on_missing_chunk(tmp_path):
    storage = ChunkStorageManager(base_staging_dir=tmp_path / "staging")
    upload_id = "test_incomplete"

    storage.write_chunk(upload_id, 0, b"Hello")
    # Chunk 1 missing
    storage.write_chunk(upload_id, 2, b"World")

    output_path = tmp_path / "failed.mp4"
    with pytest.raises(FileNotFoundError):
        storage.assemble_file(upload_id, 3, output_path)
