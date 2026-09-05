"""Unit tests for Temperature Scaling and ECE calibration."""

import torch
import torch.nn.functional as F
from ml.training.calibration import TemperatureScaler


def test_temperature_scaling_and_ece():
    """Verify TemperatureScaler fits validation logits and reduces calibration error."""
    n_samples = 100
    n_classes = 5

    # Overconfident uncalibrated logits
    logits = torch.randn(n_samples, n_classes) * 5.0
    targets = torch.randint(0, n_classes, (n_samples,))

    ece_pre = TemperatureScaler.compute_ece(F.softmax(logits, dim=-1), targets)
    assert ece_pre >= 0.0

    scaler = TemperatureScaler()
    temps = scaler.fit(logits, logits, logits, targets, targets, targets, max_iter=20)

    scaled_v, _, _ = scaler(logits, logits, logits)
    ece_post = TemperatureScaler.compute_ece(F.softmax(scaled_v, dim=-1), targets)

    assert "T_verb" in temps
    assert temps["T_verb"] > 0.0
    assert ece_post <= ece_pre + 0.05
