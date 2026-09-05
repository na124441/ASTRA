"""Unit tests for CausalTemporalActionLSTM and StaticFrameMLP models."""

import torch
from ml.activity.models.baseline import StaticFrameMLP
from ml.activity.models.lstm import CausalTemporalActionLSTM
from ml.datasets.schemas import OBJECT_VOCAB, TARGET_VOCAB, VERB_VOCAB
from ml.training.losses import MultiTaskActionLoss


def test_lstm_forward_and_backward():
    """Verify Causal LSTM forward output shapes and backward autograd gradient flow."""
    batch_size = 4
    window_size = 30
    input_dim = 26

    model = CausalTemporalActionLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2)
    criterion = MultiTaskActionLoss()

    x = torch.randn(batch_size, window_size, input_dim, requires_grad=True)
    y_v = torch.randint(0, len(VERB_VOCAB), (batch_size,))
    y_o = torch.randint(0, len(OBJECT_VOCAB), (batch_size,))
    y_t = torch.randint(0, len(TARGET_VOCAB), (batch_size,))

    v_logits, o_logits, t_logits = model(x)

    assert v_logits.shape == (batch_size, len(VERB_VOCAB))
    assert o_logits.shape == (batch_size, len(OBJECT_VOCAB))
    assert t_logits.shape == (batch_size, len(TARGET_VOCAB))

    loss, metrics = criterion(v_logits, o_logits, t_logits, y_v, y_o, y_t)
    assert loss.item() > 0.0

    loss.backward()

    # Verify input gradients flowed back properly
    assert x.grad is not None
    assert torch.norm(x.grad) > 0.0


def test_static_baseline_forward():
    """Verify StaticFrameMLP baseline executes properly on 3D or 2D inputs."""
    model = StaticFrameMLP(input_dim=26, hidden_dim=64)
    x = torch.randn(2, 30, 26)
    v, o, t = model(x)
    assert v.shape == (2, len(VERB_VOCAB))
    assert o.shape == (2, len(OBJECT_VOCAB))
    assert t.shape == (2, len(TARGET_VOCAB))
