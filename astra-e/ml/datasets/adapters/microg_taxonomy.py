"""MicroG-4M Taxonomy & Deterministic Label Mapping Manager.

Maps sparse MicroG action IDs (1, 3, 5, ..., 80 across 50 classes) to
contiguous class indices [0 .. 49] with bidirectional lookup and class weighting.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Sequence
import numpy as np

logger = logging.getLogger("astra.ml.microg_taxonomy")

# Bundled fallback pbtxt string matching official MicroG-4M release
PBTXT_FALLBACK = """
item { name: "bend/bow (at the waist)" id: 1 label_type: PERSON_MOVEMENT }
item { name: "crouch/kneel" id: 3 label_type: PERSON_MOVEMENT }
item { name: "fall down" id: 5 label_type: PERSON_MOVEMENT }
item { name: "get up" id: 6 label_type: PERSON_MOVEMENT }
item { name: "jump/leap" id: 7 label_type: PERSON_MOVEMENT }
item { name: "lie/sleep" id: 8 label_type: PERSON_MOVEMENT }
item { name: "martial art" id: 9 label_type: PERSON_MOVEMENT }
item { name: "run/jog" id: 10 label_type: PERSON_MOVEMENT }
item { name: "sit" id: 11 label_type: PERSON_MOVEMENT }
item { name: "stand" id: 12 label_type: PERSON_MOVEMENT }
item { name: "walk" id: 14 label_type: PERSON_MOVEMENT }
item { name: "carry/hold (an object)" id: 17 label_type: OBJECT_MANIPULATION }
item { name: "climb (e.g., a mountain)" id: 20 label_type: OBJECT_MANIPULATION }
item { name: "close (e.g., a door, a box)" id: 22 label_type: OBJECT_MANIPULATION }
item { name: "cut" id: 24 label_type: OBJECT_MANIPULATION }
item { name: "dress/put on clothing" id: 26 label_type: OBJECT_MANIPULATION }
item { name: "drink" id: 27 label_type: OBJECT_MANIPULATION }
item { name: "drive (e.g., a car, a truck)" id: 28 label_type: OBJECT_MANIPULATION }
item { name: "eat" id: 29 label_type: OBJECT_MANIPULATION }
item { name: "enter" id: 30 label_type: OBJECT_MANIPULATION }
item { name: "exit" id: 31 label_type: OBJECT_MANIPULATION }
item { name: "extract" id: 32 label_type: OBJECT_MANIPULATION }
item { name: "grab (a person)" id: 34 label_type: PERSON_INTERACTION }
item { name: "hit (an object)" id: 36 label_type: OBJECT_MANIPULATION }
item { name: "hug (a person)" id: 38 label_type: PERSON_INTERACTION }
item { name: "kiss (a person)" id: 41 label_type: PERSON_INTERACTION }
item { name: "lift (a person)" id: 43 label_type: PERSON_INTERACTION }
item { name: "listen to (a person)" id: 44 label_type: PERSON_INTERACTION }
item { name: "open (e.g., a window, a car door)" id: 47 label_type: OBJECT_MANIPULATION }
item { name: "play with kids" id: 49 label_type: PERSON_INTERACTION }
item { name: "point to (an object)" id: 51 label_type: OBJECT_MANIPULATION }
item { name: "pull (an object)" id: 53 label_type: OBJECT_MANIPULATION }
item { name: "push (an object)" id: 54 label_type: OBJECT_MANIPULATION }
item { name: "put down" id: 56 label_type: OBJECT_MANIPULATION }
item { name: "read" id: 57 label_type: OBJECT_MANIPULATION }
item { name: "ride (e.g., a bike, a car, a horse)" id: 58 label_type: OBJECT_MANIPULATION }
item { name: "sail boat" id: 59 label_type: OBJECT_MANIPULATION }
item { name: "shoot" id: 61 label_type: OBJECT_MANIPULATION }
item { name: "sing to (e.g., self, a person, a group)" id: 63 label_type: PERSON_INTERACTION }
item { name: "take a photo" id: 64 label_type: OBJECT_MANIPULATION }
item { name: "talk to (e.g., self, a person, a group)" id: 65 label_type: PERSON_INTERACTION }
item { name: "throw" id: 67 label_type: OBJECT_MANIPULATION }
item { name: "touch (an object)" id: 68 label_type: OBJECT_MANIPULATION }
item { name: "turn (e.g., a screwdriver)" id: 69 label_type: OBJECT_MANIPULATION }
item { name: "watch (e.g., TV)" id: 70 label_type: OBJECT_MANIPULATION }
item { name: "watch (a person)" id: 72 label_type: PERSON_INTERACTION }
item { name: "wave" id: 74 label_type: PERSON_INTERACTION }
item { name: "write" id: 76 label_type: OBJECT_MANIPULATION }
item { name: "float" id: 79 label_type: PERSON_MOVEMENT }
item { name: "tether/secure" id: 80 label_type: OBJECT_MANIPULATION }
"""


def parse_pbtxt(text: str) -> dict[int, dict[str, str]]:
    """Parse AVA-style protobuf text definition into dictionary."""
    items = re.findall(r"item\s*\{([^}]+)\}", text)
    result = {}
    for it in items:
        id_m = re.search(r"id:\s*(\d+)", it)
        name_m = re.search(r'name:\s*"([^"]+)"', it)
        type_m = re.search(r"label_type:\s*(\w+)", it)
        if id_m and name_m:
            action_id = int(id_m.group(1))
            name = name_m.group(1)
            ltype = type_m.group(1) if type_m else "UNKNOWN"
            result[action_id] = {"name": name, "label_type": ltype}
    return result


class MicroGTaxonomy:
    """
    Taxonomy and label indexer for MicroG-4M action annotations.
    Guarantees deterministic, contiguous 0..N-1 class indices.
    """

    def __init__(self, pbtxt_path: str | Path | None = None) -> None:
        self.pbtxt_info = self._load_pbtxt(pbtxt_path)
        # Deterministically sorted sparse IDs present in taxonomy
        self.sorted_action_ids = sorted(self.pbtxt_info.keys())
        self.sparse_to_contiguous: dict[int, int] = {
            act_id: idx for idx, act_id in enumerate(self.sorted_action_ids)
        }
        self.contiguous_to_sparse: dict[int, int] = {
            idx: act_id for act_id, idx in self.sparse_to_contiguous.items()
        }
        self.num_classes = len(self.sorted_action_ids)

    def _load_pbtxt(self, pbtxt_path: str | Path | None) -> dict[int, dict[str, str]]:
        if pbtxt_path and Path(pbtxt_path).exists():
            try:
                with open(pbtxt_path, "r", encoding="utf-8") as f:
                    return parse_pbtxt(f.read())
            except Exception as e:
                logger.warning("Failed reading pbtxt from %s: %s. Using fallback.", pbtxt_path, e)

        # Try Hugging Face download if possible
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id="lei-qi-233/MicroG-4M",
                filename="label_map/label_map.pbtxt",
                repo_type="dataset",
            )
            with open(downloaded, "r", encoding="utf-8") as f:
                return parse_pbtxt(f.read())
        except Exception:
            pass

        return parse_pbtxt(PBTXT_FALLBACK)

    def to_contiguous(self, sparse_action_id: int) -> int:
        """Map sparse MicroG action ID to 0..N-1 contiguous class index."""
        if sparse_action_id not in self.sparse_to_contiguous:
            raise KeyError(
                f"Unknown MicroG action ID: {sparse_action_id}. "
                f"Valid sparse IDs: {self.sorted_action_ids}"
            )
        return self.sparse_to_contiguous[sparse_action_id]

    def to_sparse(self, contiguous_class_idx: int) -> int:
        """Map contiguous class index back to original MicroG action ID."""
        if contiguous_class_idx not in self.contiguous_to_sparse:
            raise KeyError(f"Invalid class index: {contiguous_class_idx}. Valid range: 0..{self.num_classes - 1}")
        return self.contiguous_to_sparse[contiguous_class_idx]

    def get_class_name(self, class_idx: int) -> str:
        """Get human-readable action label name for a contiguous class index."""
        sparse_id = self.to_sparse(class_idx)
        return self.pbtxt_info.get(sparse_id, {}).get("name", f"Action_{sparse_id}")

    def export_label_map(self, output_path: str | Path) -> Path:
        """
        Export label map to JSON matching the required specification:
        {
          "1": {
            "class_index": 0,
            "name": "bend/bow (at the waist)",
            "label_type": "PERSON_MOVEMENT"
          }, ...
        }
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        mapping = {}
        for sparse_id, class_idx in self.sparse_to_contiguous.items():
            info = self.pbtxt_info.get(sparse_id, {"name": f"Action_{sparse_id}", "label_type": "UNKNOWN"})
            mapping[str(sparse_id)] = {
                "class_index": class_idx,
                "name": info["name"],
                "label_type": info.get("label_type", "UNKNOWN"),
            }

        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)

        return out_p

    def compute_class_weights(
        self,
        class_indices: Sequence[int],
        smoothing: float = 0.1,
    ) -> np.ndarray:
        """
        Computes inverse frequency class weights to balance CrossEntropyLoss.
        Weights normalized to mean = 1.0.
        """
        counts = np.zeros(self.num_classes, dtype=np.float32)
        for idx in class_indices:
            if 0 <= idx < self.num_classes:
                counts[idx] += 1.0

        total_samples = float(len(class_indices))
        if total_samples == 0:
            return np.ones(self.num_classes, dtype=np.float32)

        smoothed_counts = counts + (smoothing * total_samples / self.num_classes)
        weights = total_samples / (self.num_classes * smoothed_counts)
        weights = weights / np.mean(weights)
        return weights


def load_microg_actions_csv(csv_path: str | Path | None = None) -> list[dict[str, Any]]:
    """
    Load actions annotation CSV rows from local path, Hugging Face download,
    or datasets library.
    """
    if csv_path and Path(csv_path).exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    # Attempt Hugging Face datasets load
    try:
        from datasets import load_dataset
        ds = load_dataset("lei-qi-233/MicroG-4M", "actions", split="full")
        rows = [
            {
                "video_id": item["video_id"],
                "movie_or_real": item.get("movie_or_real", ""),
                "person_id": str(item.get("person_id", "1")),
                "action": str(item["action"]),
            }
            for item in ds
        ]
        return rows
    except Exception:
        pass

    # Fallback to direct file download via huggingface_hub
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="lei-qi-233/MicroG-4M",
            filename="annotation_files/actions.csv",
            repo_type="dataset",
        )
        with open(downloaded, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        logger.error("Failed downloading MicroG actions.csv: %s", e)
        raise FileNotFoundError(
            "Could not access MicroG-4M actions.csv from Hugging Face or local path. "
            "Ensure internet access or provide --actions-csv <PATH>."
        ) from e
