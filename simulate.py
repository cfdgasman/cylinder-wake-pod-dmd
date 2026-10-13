"""Run the lattice-Boltzmann cylinder simulation and store vorticity snapshots in data/wake.npz."""

import time
from pathlib import Path

import numpy as np

from wake.lbm import Cylinder, run

STEPS = 40_000
SAMPLE_FROM = 28_000
SAMPLE_EVERY = 25


def main():
    case = Cylinder()
    t0 = time.perf_counter()
    snaps, times, probe_v = run(case, STEPS, SAMPLE_EVERY, SAMPLE_FROM)
    print(f"{STEPS} steps in {time.perf_counter() - t0:.0f} s, {len(snaps)} snapshots")
    Path("data").mkdir(exist_ok=True)
    np.savez_compressed(
        "data/wake.npz", vorticity=snaps, steps=times, probe_v=probe_v, mask=case.mask(),
        diameter=case.diameter, u_in=case.u_in, re=case.re, cx=case.cx, cy=case.cy,
    )


if __name__ == "__main__":
    main()
