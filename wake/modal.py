"""Modal decompositions of snapshot data: POD (via the SVD) and exact DMD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class POD:
    mean: np.ndarray  # (m,)
    modes: np.ndarray  # (m, r) orthonormal spatial modes
    sigma: np.ndarray  # (r,) singular values
    coeffs: np.ndarray  # (r, nt) temporal coefficients = sigma * V^T

    @property
    def energy(self):
        e = self.sigma**2
        return e / e.sum()

    def reconstruct(self, rank):
        return self.mean[:, None] + self.modes[:, :rank] @ self.coeffs[:rank]


def pod(X):
    """POD of snapshot matrix X (m, nt): mean-subtracted thin SVD."""
    mean = X.mean(axis=1)
    U, s, Vh = np.linalg.svd(X - mean[:, None], full_matrices=False)
    return POD(mean, U, s, s[:, None] * Vh)


@dataclass
class DMD:
    eigs: np.ndarray  # (r,) discrete-time eigenvalues lambda
    modes: np.ndarray  # (m, r) exact DMD modes
    amplitudes: np.ndarray  # (r,)
    dt: float

    @property
    def omega(self):
        """Continuous-time eigenvalues log(lambda) / dt."""
        return np.log(self.eigs.astype(complex)) / self.dt

    @property
    def frequency(self):
        return self.omega.imag / (2 * np.pi)

    @property
    def growth(self):
        return self.omega.real

    def reconstruct(self, nt):
        k = np.arange(nt)
        dynamics = self.amplitudes[:, None] * self.eigs[:, None] ** k[None, :]
        return (self.modes @ dynamics).real


def dmd(X, rank, dt=1.0):
    """Exact DMD (Tu et al. 2014) of snapshots X (m, nt) with SVD truncation `rank`."""
    X1, X2 = X[:, :-1], X[:, 1:]
    U, s, Vh = np.linalg.svd(X1, full_matrices=False)
    U, s, V = U[:, :rank], s[:rank], Vh[:rank].conj().T
    Atilde = U.conj().T @ X2 @ V / s
    eigs, W = np.linalg.eig(Atilde)
    Phi = X2 @ V / s @ W
    b = np.linalg.lstsq(Phi, X[:, 0], rcond=None)[0]
    return DMD(eigs, Phi, b, dt)


def dominant_frequency(signal, dt=1.0, pad=16):
    """Peak frequency of a signal from a zero-padded, Hann-windowed FFT."""
    x = (signal - signal.mean()) * np.hanning(len(signal))
    n = pad * len(x)
    spec = np.abs(np.fft.rfft(x, n))
    freqs = np.fft.rfftfreq(n, dt)
    return freqs[np.argmax(spec[1:]) + 1], freqs, spec
