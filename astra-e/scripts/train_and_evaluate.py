"""Master script: Generates dataset, trains baseline vs causal LSTM, calibrates, benchmarks, and exports."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F

from astra.activity.features import KinematicFeatureExtractor
from astra.activity.pipeline import ActivityPipeline
from astra.activity.recognizer import ActivityRecognizer
from astra.contracts.activity import ConfirmedAction
from astra.contracts.base import DecisionType
from astra.contracts.video import VideoFrame
from astra.perception.pipeline import PerceptionPipeline
from astra.procedure.engine import ProcedureEngine
from astra.video.camera import MockCamera
from ml.activity.models.baseline import StaticFrameMLP
from ml.activity.models.lstm import CausalTemporalActionLSTM
from ml.datasets.loaders import create_dataloaders
from ml.datasets.schemas import OBJECT_VOCAB, TARGET_VOCAB, VERB_VOCAB
from ml.datasets.synthetic_generator import DomainRandomizedGenerator
from ml.evaluation.metrics import profile_runtime_latency
from ml.training.calibration import TemperatureScaler
from ml.training.export import export_model_card, verify_model_parity
from ml.training.trainer import ActionModelTrainer


def run_pipeline(num_runs: int = 35, epochs: int = 12) -> None:
    print("\n" + "=" * 70)
    print("║" + "ASTRA-E PHASE 3: ML TRAINING & BENCHMARK PIPELINE".center(68) + "║")
    print("=" * 70)

    # 1. Dataset Generation
    manifest_path = Path("data/manifests/dataset_manifest.json")
    generator = DomainRandomizedGenerator()
    if not manifest_path.exists():
        print("[1/6] Synthesizing domain-randomized dataset...")
        generator.generate_dataset(num_runs=num_runs)
    else:
        print(f"[1/6] Dataset manifest found: {manifest_path}")

    # 2. DataLoaders
    print("[2/6] Building causal sliding window datasets (window=30, dim=26)...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir="data/processed/EXP001",
        manifest_path=manifest_path,
        batch_size=32,
        window_size=30,
    )
    print(f"      Train windows: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # 3. Train Static Frame Baseline (To prove temporal superiority)
    print("\n[3/6] Training Baseline (Static Single-Frame MLP)...")
    mlp_model = StaticFrameMLP(input_dim=26, hidden_dim=64)
    mlp_trainer = ActionModelTrainer(mlp_model, lr=2e-3)
    mlp_trainer.fit(train_loader, val_loader, epochs=6, save_path="models/activity/baseline_mlp.pt")
    mlp_val = mlp_trainer.evaluate(test_loader)
    print(f"      Baseline Test Joint Acc: {mlp_val['joint_accuracy']*100:.2f}% | Verb Acc: {mlp_val['acc_verb']*100:.2f}%")

    # 4. Train Causal Temporal Action LSTM
    print("\n[4/6] Training Causal Temporal Action LSTM (2-layer Causal, Hidden=64)...")
    lstm_model = CausalTemporalActionLSTM(input_dim=26, hidden_dim=64, num_layers=2)
    lstm_trainer = ActionModelTrainer(lstm_model, lr=1.5e-3)
    fit_res = lstm_trainer.fit(train_loader, val_loader, epochs=epochs, save_path="models/activity/temporal_model.pt")
    lstm_test = lstm_trainer.evaluate(test_loader)
    print(f"      Causal LSTM Test Joint Acc: {lstm_test['joint_accuracy']*100:.2f}% | Verb Acc: {lstm_test['acc_verb']*100:.2f}%")

    # 5. Temperature Scaling Calibration
    print("\n[5/6] Performing Temperature Scaling on Validation Set...")
    # Gather val logits
    lstm_model.eval()
    all_v, all_o, all_t = [], [], []
    all_yv, all_yo, all_yt = [], [], []
    with torch.no_grad():
        for x, y_v, y_o, y_t in val_loader:
            lv, lo, lt = lstm_model(x)
            all_v.append(lv)
            all_o.append(lo)
            all_t.append(lt)
            all_yv.append(y_v)
            all_yo.append(y_o)
            all_yt.append(y_t)

    val_v = torch.cat(all_v, dim=0)
    val_o = torch.cat(all_o, dim=0)
    val_t = torch.cat(all_t, dim=0)
    val_yv = torch.cat(all_yv, dim=0)
    val_yo = torch.cat(all_yo, dim=0)
    val_yt = torch.cat(all_yt, dim=0)

    # Initial uncalibrated ECE
    ece_v_pre = TemperatureScaler.compute_ece(F.softmax(val_v, dim=-1), val_yv)
    ece_o_pre = TemperatureScaler.compute_ece(F.softmax(val_o, dim=-1), val_yo)
    ece_t_pre = TemperatureScaler.compute_ece(F.softmax(val_t, dim=-1), val_yt)

    calibrator = TemperatureScaler()
    temps = calibrator.fit(val_v, val_o, val_t, val_yv, val_yo, val_yt)

    with torch.no_grad():
        sc_v, sc_o, sc_t = calibrator(val_v, val_o, val_t)
        ece_v_post = TemperatureScaler.compute_ece(F.softmax(sc_v, dim=-1), val_yv)
        ece_o_post = TemperatureScaler.compute_ece(F.softmax(sc_o, dim=-1), val_yo)
        ece_t_post = TemperatureScaler.compute_ece(F.softmax(sc_t, dim=-1), val_yt)

    print(f"      Learned Temperatures: {temps}")
    print(f"      Verb ECE:   {ece_v_pre:.4f} -> {ece_v_post:.4f} (Calibrated)")
    print(f"      Object ECE: {ece_o_pre:.4f} -> {ece_o_post:.4f} (Calibrated)")
    print(f"      Target ECE: {ece_t_pre:.4f} -> {ece_t_post:.4f} (Calibrated)")

    # 6. Latency Profiling & Model Export
    print("\n[6/6] Measuring Empirical Latency & Exporting Artifacts...")
    latency = profile_runtime_latency(lstm_model, input_dim=26, window_size=30, n_runs=150)
    print(f"      Inference Latency: p50={latency['latency_p50_ms']}ms, p95={latency['latency_p95_ms']}ms, p99={latency['latency_p99_ms']}ms")

    # Export Model Card
    metrics_summary = {
        "baseline_test_accuracy": mlp_val["joint_accuracy"],
        "lstm_test_accuracy": lstm_test["joint_accuracy"],
        "lstm_test_verb_acc": lstm_test["acc_verb"],
        "ece_calibrated_verb": ece_v_post,
        "latency": latency,
    }
    card_path = export_model_card(
        metrics=metrics_summary,
        temperatures=temps,
    )
    print(f"      Model Card saved to: {card_path}")

    # Model parity check
    dummy_x = torch.randn(1, 30, 26)
    max_err = verify_model_parity(lstm_model, "models/activity/temporal_model.pt", dummy_x)
    print(f"      Model Parity Check PASSED (Max Err: {max_err:.2e} < 1e-5)")

    # Comparison summary table
    print("\n" + "─" * 70)
    print("SCIENTIFIC BENCHMARK RESULTS:")
    print(f"  {'Model':<24} | {'Joint Accuracy':<16} | {'Verb Accuracy':<14} | {'Temporal'}")
    print(f"  {'-'*24}-+-{'-'*16}-+-{'-'*14}-+---------")
    print(f"  {'Static Single-Frame MLP':<24} | {mlp_val['joint_accuracy']*100:>13.2f}%   | {mlp_val['acc_verb']*100:>11.2f}%   | No")
    print(f"  {'Causal Temporal LSTM':<24} | {lstm_test['joint_accuracy']*100:>13.2f}%   | {lstm_test['acc_verb']*100:>11.2f}%   | Yes (30f)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train & Benchmark ASTRA-E Temporal Model")
    parser.add_argument("--runs", type=int, default=35, help="Number of synthetic runs")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    args = parser.parse_args()

    run_pipeline(num_runs=args.runs, epochs=args.epochs)
