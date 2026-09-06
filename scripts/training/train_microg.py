"""Main Training Entrypoint for MicroG-4M Causal Temporal Baseline.

Reproducible baseline for Google Colab GPU execution.
Implements:
- MicroG actions taxonomy mapping (50 classes: 0..49)
- Deterministic grouped video-level splitting (zero temporal leakage)
- Causal 30-frame sliding window representations
- Causal unidirectional LSTM action classifier
- Class-weighted CrossEntropyLoss & macro-F1 tracking
- Best checkpoint serialization & test set evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path (supports running from repo root or astra-e subdir)
current_dir = Path(__file__).resolve().parent
for candidate in [
    current_dir.parent.parent,
    current_dir.parent.parent / "astra-e",
    current_dir.parent.parent.parent,
    current_dir.parent.parent.parent / "astra-e",
]:
    if (candidate / "ml").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.activity.models.microg_lstm import CausalMicroGLSTM
from ml.datasets.adapters.microg_splitter import MicroGGroupedSplitter
from ml.datasets.adapters.microg_taxonomy import MicroGTaxonomy, load_microg_actions_csv
from ml.datasets.adapters.microg_temporal_dataset import (
    MicroGTemporalDataset,
    MicroGVideoUnavailableError,
)
from ml.training.microg_metrics import MicroGMetricsTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("astra.train_microg")


def set_seed(seed: int) -> None:
    """Enforce full determinism across all random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_startup_banner(
    device: torch.device,
    dataset: str,
    config: str,
    classes: int,
    window_size: int,
    batch_size: int,
) -> None:
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A (CPU Mode)"
    print("=" * 60)
    print("ASTRA-E MicroG-4M Temporal Baseline Training")
    print(f"Device: {device.type}")
    print(f"GPU: {gpu_name}")
    print(f"Dataset: {dataset}")
    print(f"Configuration: {config}")
    print(f"Classes: {classes}")
    print(f"Window size: {window_size}")
    print(f"Batch size: {batch_size}")
    print("=" * 60)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += float(loss.item()) * len(y)
        preds = torch.argmax(logits, dim=-1)
        correct += int((preds == y).sum().item())
        total += len(y)

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    metrics_tracker: MicroGMetricsTracker,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    metrics_tracker.reset()
    total_loss = 0.0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * len(y)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        targets = y.cpu().numpy()

        metrics_tracker.update(preds, targets)
        total += len(y)

    avg_loss = total_loss / max(1, total)
    results = metrics_tracker.compute()
    return avg_loss, results


def run_training(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_startup_banner(
        device=device,
        dataset=args.dataset,
        config=args.config,
        classes=50,
        window_size=args.window_size,
        batch_size=args.batch_size,
    )

    # -------------------------------------------------------------------------
    # 1. Taxonomy & Label Map
    # -------------------------------------------------------------------------
    logger.info("Initializing MicroG taxonomy and deterministic label mapping...")
    taxonomy = MicroGTaxonomy(pbtxt_path=args.pbtxt)
    label_map_file = output_dir / "label_map.json"
    taxonomy.export_label_map(label_map_file)
    logger.info("Saved contiguous label map (50 classes) to: %s", label_map_file)

    # -------------------------------------------------------------------------
    # 2. Annotations & Reality Check
    # -------------------------------------------------------------------------
    logger.info("Loading MicroG-4M '%s' annotations...", args.config)
    try:
        annotations = load_microg_actions_csv(args.actions_csv)
    except Exception as e:
        logger.error("Could not load annotations: %s", e)
        return 1

    logger.info("Loaded %d annotation rows across %d classes.", len(annotations), taxonomy.num_classes)

    # -------------------------------------------------------------------------
    # 3. Deterministic Grouped Split (Zero Leakage)
    # -------------------------------------------------------------------------
    logger.info("Executing grouped video-level split (70%% train / 15%% val / 15%% test, seed=%d)...", args.seed)
    splitter = MicroGGroupedSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=args.seed)
    partitioned = splitter.split_annotations(annotations, group_by="video_id")

    split_manifest_file = output_dir / "split_manifest.json"
    splitter.export_split_manifest(partitioned, split_manifest_file, group_by="video_id")
    logger.info(
        "Split counts: Train=%d samples (%d videos), Val=%d samples (%d videos), Test=%d samples (%d videos)",
        len(partitioned["train"]),
        len(set(r["video_id"] for r in partitioned["train"])),
        len(partitioned["val"]),
        len(set(r["video_id"] for r in partitioned["val"])),
        len(partitioned["test"]),
        len(set(r["video_id"] for r in partitioned["test"])),
    )
    logger.info("Saved split manifest to: %s", split_manifest_file)

    # -------------------------------------------------------------------------
    # 4. Data Availability Reality Check (Section 4)
    # -------------------------------------------------------------------------
    video_dir = Path(args.video_dir) if args.video_dir else None
    feature_dir = Path(args.feature_dir) if args.feature_dir else None

    if not args.dry_run and not video_dir and not feature_dir:
        # Fails closed with descriptive actionable report
        err_msg = (
            "\n" + "=" * 80 + "\n"
            "[CRITICAL DATASET REALITY CHECK] MicroG-4M Video Source Not Found!\n\n"
            "The Hugging Face dataset 'lei-qi-233/MicroG-4M' (config='actions') provides\n"
            "annotation metadata (actions.csv, bounding_boxes.csv, label_map.pbtxt), but\n"
            "does NOT host raw MP4 video files directly due to licensing and copyright\n"
            "constraints (as documented in MicroG-4M README.md and video_id_list.pdf).\n\n"
            "In accordance with ASTRA-E scientific integrity requirements:\n"
            "  - Synthetic or random video frames will NOT be fabricated.\n"
            "  - The model will not falsely claim to have trained on MicroG video.\n"
            "  - Training cannot proceed without authentic video files or feature representations.\n\n"
            "To train the real temporal baseline on Google Colab or workstation:\n"
            "  1. Download the corresponding MicroG video clips (see videos/video_id_list.pdf)\n"
            "     and place them in a folder, e.g. data/microg/videos/<video_id>.mp4\n"
            "  2. Run training with --video-dir:\n"
            "     python scripts/training/train_microg.py --video-dir data/microg/videos\n"
            "  3. Or provide pre-extracted feature representations with --feature-dir:\n"
            "     python scripts/training/train_microg.py --feature-dir data/microg/features\n"
            "  4. To test pipeline mechanics and model serialization in CI, pass --dry-run.\n"
            "=" * 80
        )
        print(err_msg, file=sys.stderr)
        return 1

    # -------------------------------------------------------------------------
    # 5. Datasets & Loaders
    # -------------------------------------------------------------------------
    try:
        train_ds = MicroGTemporalDataset(
            annotations=partitioned["train"] if not args.dry_run else partitioned["train"][:64],
            taxonomy=taxonomy,
            video_dir=video_dir,
            feature_dir=feature_dir,
            window_size=args.window_size,
            feature_dim=args.feature_dim,
            allow_synthetic_test=args.dry_run,
        )
        val_ds = MicroGTemporalDataset(
            annotations=partitioned["val"] if not args.dry_run else partitioned["val"][:32],
            taxonomy=taxonomy,
            video_dir=video_dir,
            feature_dir=feature_dir,
            window_size=args.window_size,
            feature_dim=args.feature_dim,
            allow_synthetic_test=args.dry_run,
        )
        test_ds = MicroGTemporalDataset(
            annotations=partitioned["test"] if not args.dry_run else partitioned["test"][:32],
            taxonomy=taxonomy,
            video_dir=video_dir,
            feature_dir=feature_dir,
            window_size=args.window_size,
            feature_dim=args.feature_dim,
            allow_synthetic_test=args.dry_run,
        )
    except MicroGVideoUnavailableError as e:
        print(str(e), file=sys.stderr)
        return 1

    use_pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if not args.dry_run else 0,
        pin_memory=use_pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if not args.dry_run else 0,
        pin_memory=use_pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if not args.dry_run else 0,
        pin_memory=use_pin,
    )

    # -------------------------------------------------------------------------
    # 6. Model, Loss & Optimizer
    # -------------------------------------------------------------------------
    model = CausalMicroGLSTM(
        input_dim=args.feature_dim,
        hidden_dim=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=taxonomy.num_classes,
        dropout=args.dropout,
    ).to(device)

    # Class weighting
    if args.use_class_weights:
        train_class_indices = [s["class_idx"] for s in train_ds.samples]
        weights = taxonomy.compute_class_weights(train_class_indices)
        weight_tensor = torch.from_numpy(weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        logger.info("Using class-weighted CrossEntropyLoss.")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    metrics_tracker = MicroGMetricsTracker(num_classes=taxonomy.num_classes, taxonomy=taxonomy)

    # -------------------------------------------------------------------------
    # 7. Training Loop
    # -------------------------------------------------------------------------
    best_val_f1 = -1.0
    best_epoch = 0
    best_checkpoint_path = output_dir / "best.pt"

    epochs_to_run = args.epochs if not args.dry_run else 1
    logger.info("Starting training loop (%d epochs)...", epochs_to_run)

    for epoch in range(1, epochs_to_run + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
        )
        val_loss, val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            metrics_tracker=metrics_tracker,
            device=device,
        )
        elapsed = time.time() - t0

        logger.info(
            "Epoch [%d/%d] (%.1fs) | Train Loss: %.4f, Acc: %.2f%% | Val Loss: %.4f, Acc: %.2f%%, Macro-F1: %.4f",
            epoch,
            epochs_to_run,
            elapsed,
            train_loss,
            train_acc * 100.0,
            val_loss,
            val_metrics["accuracy"] * 100.0,
            val_metrics["macro_f1"],
        )

        # Checkpoint based on validation Macro-F1
        if val_metrics["macro_f1"] > best_val_f1 or epoch == 1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_macro_f1": best_val_f1,
                "best_val_accuracy": val_metrics["accuracy"],
                "class_mapping": taxonomy.sparse_to_contiguous,
                "num_classes": taxonomy.num_classes,
                "model_config": {
                    "input_dim": args.feature_dim,
                    "hidden_dim": args.hidden_size,
                    "num_layers": args.num_layers,
                    "num_classes": taxonomy.num_classes,
                    "dropout": args.dropout,
                    "window_size": args.window_size,
                },
                "random_seed": args.seed,
            }
            torch.save(checkpoint_data, best_checkpoint_path)
            logger.info("  -> Saved best checkpoint (Val Macro-F1: %.4f) to %s", best_val_f1, best_checkpoint_path)

    # -------------------------------------------------------------------------
    # 8. Final Test Set Evaluation
    # -------------------------------------------------------------------------
    logger.info("Loading best checkpoint from epoch %d for final test evaluation...", best_epoch)
    if best_checkpoint_path.exists():
        ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        metrics_tracker=metrics_tracker,
        device=device,
    )

    # Print required Section 11 banner
    final_banner = metrics_tracker.format_final_report(test_metrics, best_epoch=best_epoch)
    print("\n" + final_banner + "\n")

    # Export metrics and classification report
    metrics_file, report_file = metrics_tracker.export_reports(
        metrics=test_metrics,
        output_dir=output_dir,
        best_epoch=best_epoch,
    )
    logger.info("Exported metrics to: %s", metrics_file)
    logger.info("Exported per-class report to: %s", report_file)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ASTRA-E MicroG-4M Causal Temporal Baseline Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="lei-qi-233/MicroG-4M", help="Hugging Face dataset identifier")
    parser.add_argument("--config", default="actions", help="MicroG-4M dataset configuration")
    parser.add_argument("--actions-csv", default=None, help="Optional local actions.csv path")
    parser.add_argument("--pbtxt", default=None, help="Optional local label_map.pbtxt path")
    parser.add_argument("--video-dir", default=None, help="Path to directory containing MicroG MP4 videos")
    parser.add_argument("--feature-dir", default=None, help="Path to directory containing pre-extracted feature arrays")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW")
    parser.add_argument("--window-size", type=int, default=30, help="Causal temporal sliding window size (frames)")
    parser.add_argument("--feature-dim", type=int, default=128, help="Per-frame visual feature dimension")
    parser.add_argument("--hidden-size", type=int, default=256, help="LSTM hidden layer size")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", default="outputs/microg_baseline", help="Directory for artifacts & logs")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--use-class-weights", action="store_true", default=True, help="Enable class-weighted CrossEntropyLoss")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Run 1 synthetic test epoch for pipeline mechanics verification")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(run_training(args))
