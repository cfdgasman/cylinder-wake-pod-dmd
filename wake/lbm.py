"""D2Q9 lattice-Boltzmann (BGK) solver for 2D flow past a circular cylinder.

* Lattice units: dx = dt = 1, c_s^2 = 1/3, nu = (tau - 1/2) / 3
* Inlet (x = 0): prescribed velocity through the equilibrium distribution
* Outlet (x = nx-1): zero-gradient copy of the unknown populations
* Top / bottom: periodic (the domain is wide enough that blockage stays moderate)
* Cylinder: full-way bounce-back (no slip)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])


def equilibrium(rho, ux, uy, out=None):
    usq = 1.5 * (ux * ux + uy * uy)
    if out is None:
        out = np.empty((9,) + np.shape(rho))
    for i in range(9):
        cu = 3.0 * (C[i, 0] * ux + C[i, 1] * uy)
        out[i] = W[i] * rho * (1.0 + cu + 0.5 * cu * cu - usq)
    return out


@dataclass
class Cylinder:
    nx: int = 400
    ny: int = 120
    diameter: float = 20.0
    u_in: float = 0.08
    re: float = 100.0
    cx: float = 80.0
    cy: float = 62.0  # slightly off-centre, which breaks the symmetry and starts shedding

    @property
    def nu(self):
        return self.u_in * self.diameter / self.re

    @property
    def tau(self):
        return 3.0 * self.nu + 0.5

    def mask(self):
        x, y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= (self.diameter / 2) ** 2


def macroscopic(f):
    rho = f.sum(axis=0)
    ux = (f[1] + f[5] + f[8] - f[3] - f[6] - f[7]) / rho
    uy = (f[2] + f[5] + f[6] - f[4] - f[7] - f[8]) / rho
    return rho, ux, uy


def vorticity(ux, uy):
    return np.gradient(uy, axis=0) - np.gradient(ux, axis=1)


def run(case: Cylinder, steps: int, sample_every: int, sample_from: int, probe=(160, 60), perturb=True):
    """Advance the flow. Returns sampled vorticity snapshots (nt, nx, ny), sample times and a
    probe time series of the cross-stream velocity (every step)."""
    solid = case.mask()
    ones = np.ones((case.nx, case.ny))
    ux0 = case.u_in * ones
    uy0 = np.zeros_like(ux0)
    if perturb:
        # a transverse velocity blob in the near wake seeds the (antisymmetric) shedding mode;
        # without it the unstable steady wake can persist for a very long time
        x, y = np.meshgrid(np.arange(case.nx), np.arange(case.ny), indexing="ij")
        r2 = (x - case.cx - 2 * case.diameter) ** 2 + (y - case.cy) ** 2
        uy0 = 0.3 * case.u_in * np.exp(-r2 / (2 * case.diameter**2))
    f = equilibrium(ones, ux0, uy0)
    f_in = equilibrium(np.ones((1, case.ny)), case.u_in * np.ones((1, case.ny)), np.zeros((1, case.ny)))
    omega = 1.0 / case.tau

    snaps, times, probe_v = [], [], np.empty(steps)
    feq = np.empty_like(f)
    for n in range(steps):
        rho, ux, uy = macroscopic(f)
        ux[solid] = 0.0
        uy[solid] = 0.0
        probe_v[n] = uy[probe]
        if n >= sample_from and (n - sample_from) % sample_every == 0:
            snaps.append(vorticity(ux, uy).astype(np.float32))
            times.append(n)

        # collision (BGK), with bounce-back inside the cylinder
        equilibrium(rho, ux, uy, out=feq)
        f_post = f + omega * (feq - f)
        f_post[:, solid] = f[:, solid][OPP]

        # streaming (periodic in y; x wraps are overwritten by the boundary conditions)
        for i in range(9):
            f[i] = np.roll(f_post[i], (C[i, 0], C[i, 1]), axis=(0, 1))

        # inlet: equilibrium at the prescribed velocity; outlet: zero gradient
        f[:, 0, :] = f_in[:, 0, :]
        f[[3, 6, 7], -1, :] = f[[3, 6, 7], -2, :]
    return np.array(snaps), np.array(times), probe_v
