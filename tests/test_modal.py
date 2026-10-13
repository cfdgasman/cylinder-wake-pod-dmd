"""Checks of POD and DMD on synthetic data with known structure (no simulation needed)."""

import numpy as np
import pytest

from wake import dmd, dominant_frequency, pod


def travelling_waves(nt=200, dt=0.1):
    """Two travelling waves with known frequencies and a decaying one, on a 1D grid."""
    x = np.linspace(0, 2 * np.pi, 300)[:, None]
    t = np.arange(nt)[None, :] * dt
    X = (
        1.0
        + np.cos(2 * x - 2 * np.pi * 0.7 * t)
        + 0.5 * np.cos(5 * x - 2 * np.pi * 1.9 * t)
        + 0.3 * np.exp(-0.2 * t) * np.sin(3 * x)
    )
    return X, dt


def test_pod_is_optimal_and_orthonormal():
    X, _ = travelling_waves()
    p = pod(X)
    r = 4
    assert np.allclose(p.modes[:, :r].T @ p.modes[:, :r], np.eye(r), atol=1e-10)
    err = np.linalg.norm(X - p.reconstruct(r))
    # Eckart-Young: the rank-r error equals the norm of the discarded singular values
    assert err == pytest.approx(np.sqrt(np.sum(p.sigma[r:] ** 2)), rel=1e-8)


def test_dmd_recovers_frequencies_and_growth_rates():
    X, dt = travelling_waves()
    d = dmd(X, rank=7, dt=dt)
    f = np.sort(np.abs(d.frequency))
    assert np.any(np.isclose(f, 0.7, atol=1e-6))
    assert np.any(np.isclose(f, 1.9, atol=1e-6))
    decaying = np.isclose(np.abs(d.frequency), 0.0, atol=1e-8) & (d.growth < -0.1)
    assert np.isclose(d.growth[decaying], -0.2, atol=1e-6).any()
    assert np.linalg.norm(d.reconstruct(X.shape[1]) - X) / np.linalg.norm(X) < 1e-8


def test_dominant_frequency():
    t = np.arange(4000) * 0.01
    f, _, _ = dominant_frequency(np.sin(2 * np.pi * 3.3 * t), dt=0.01)
    assert f == pytest.approx(3.3, abs=5e-3)
