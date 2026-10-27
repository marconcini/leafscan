import numpy as np
import pytest

from leafscan.reliability import (
    accuracy_at_coverage,
    ece,
    energy,
    fit_temperature,
    lab_threshold,
    ood_auroc,
    risk_coverage,
    softmax,
)


def test_ece_zero_when_confidence_matches_accuracy():
    # 10 predictions at confidence 0.8, 8 of them right -> perfectly calibrated
    probs = np.tile([0.8, 0.2], (10, 1))
    labels = np.array([0] * 8 + [1] * 2)
    assert ece(probs, labels) == pytest.approx(0.0, abs=1e-9)
    # all right at confidence 0.8 -> underconfident by 0.2
    assert ece(probs, np.zeros(10, int)) == pytest.approx(0.2)


def test_fit_temperature_recovers_the_true_scale():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(4000, 5)) * 3
    labels = np.array([rng.choice(5, p=p) for p in softmax(true_logits)])
    t = fit_temperature(true_logits * 2.5, labels)  # model 2.5x overconfident
    assert t == pytest.approx(2.5, rel=0.1)


def test_risk_coverage_and_accuracy_at_coverage():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, True, False, False])
    cov, acc = risk_coverage(conf, correct)
    assert list(cov) == [0.25, 0.5, 0.75, 1.0]
    assert list(acc) == [1.0, 1.0, 2 / 3, 0.5]
    assert accuracy_at_coverage(conf, correct, 0.5) == 1.0


def test_lab_threshold():
    conf = np.array([0.99, 0.95, 0.9, 0.8])
    correct = np.array([True, True, True, False])
    assert lab_threshold(conf, correct, 1.0) == 0.9
    assert lab_threshold(conf, ~correct, 1.0) == 1.0  # target unreachable -> never answer


def test_ood_scores():
    id_logits = np.array([[5.0, 0.0], [6.0, 0.0]])
    ood_logits = np.array([[0.1, 0.0], [0.0, 0.2]])
    assert ood_auroc(energy(id_logits), energy(ood_logits)) == 1.0
    assert ood_auroc(np.array([0.1]), np.array([0.9])) == 0.0
