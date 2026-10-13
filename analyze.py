"""POD and DMD of the cylinder wake stored in data/wake.npz; writes figures and a GIF to docs/."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from wake import dmd, dominant_frequency, pod

X0 = 60  # analyse the wake region x >= X0 (lattice units)


def load():
    d = np.load("data/wake.npz")
    return {k: d[k] for k in d.files}


def field_axes(ax, data, extent):
    ax.add_patch(Circle((data["cx"], data["cy"]), data["diameter"] / 2, color="#444444", zorder=3))
    ax.set(xlim=extent[:2], ylim=extent[2:], xticks=[], yticks=[], aspect="equal")


def show(ax, field, data, vmax, cmap="RdBu_r"):
    nx, ny = field.shape
    extent = (X0, X0 + nx, 0, ny)
    im = ax.imshow(field.T, origin="lower", extent=extent, cmap=cmap, vmin=-vmax, vmax=vmax)
    field_axes(ax, data, extent)
    return im


def main():
    data = load()
    D, U = float(data["diameter"]), float(data["u_in"])
    vort = data["vorticity"][:, X0:, :]  # (nt, nx, ny)
    nt, nx, ny = vort.shape
    dt = float(np.diff(data["steps"])[0])  # lattice steps between snapshots
    X = vort.reshape(nt, -1).T.astype(np.float64)  # (m, nt)
    to_st = D / U  # frequency (1/step) -> Strouhal number

    # ---- reference frequency from the probe signal (every step, last 12000 steps)
    probe = data["probe_v"][-12_000:]
    f_probe, freqs, spec = dominant_frequency(probe, dt=1.0)
    st_probe = f_probe * to_st

    # ---- POD
    p = pod(X)
    cum = np.cumsum(p.energy)

    # ---- DMD
    rank = 21
    d = dmd(X, rank, dt=dt)
    order = np.argsort(-np.abs(d.amplitudes))
    st_dmd = np.abs(d.frequency) * to_st
    pos = d.frequency > 1e-12
    k1 = np.argmax(np.where(pos, np.abs(d.amplitudes), 0))
    st_1 = st_dmd[k1]

    print(f"snapshots: {nt}, spatial points: {X.shape[0]:,}, dt = {dt:.0f} steps")
    print(f"Strouhal from probe FFT:   {st_probe:.4f}")
    print(f"Strouhal from DMD mode 1:  {st_1:.4f}  (|lambda| = {abs(d.eigs[k1]):.6f})")
    print("DMD harmonics (St / St_1):", " ".join(f"{s / st_1:.3f}" for s in sorted(set(np.round(st_dmd[pos], 5)))))
    print("POD energy of first 2 / 4 / 6 / 10 modes: " + " / ".join(f"{100 * cum[k - 1]:.2f} %" for k in (2, 4, 6, 10)))

    ranks = [1, 2, 4, 6, 8, 10, 14, 20]
    Xn = np.linalg.norm(X - p.mean[:, None])
    e_pod = [np.linalg.norm(X - p.reconstruct(r)) / Xn for r in ranks]
    e_dmd = []
    for r in ranks:
        dr = dmd(X, r + 1, dt=dt)  # +1 for the mean (zero-frequency) mode
        e_dmd.append(np.linalg.norm(X - dr.reconstruct(nt)) / Xn)
    print("| rank | POD error | DMD error |\n|---|---|---|")
    for r, a, b in zip(ranks, e_pod, e_dmd):
        print(f"| {r} | {a:.2e} | {b:.2e} |")

    vmax = np.percentile(np.abs(vort), 99.5)

    # ---- figure 1: snapshot + probe spectrum
    fig = plt.figure(figsize=(12, 3.8))
    ax = fig.add_axes([0.02, 0.1, 0.62, 0.8])
    show(ax, vort[-1], data, vmax)
    ax.set_title(f"Vorticity, Re = {float(data['re']):.0f} (lattice Boltzmann, D = {D:.0f} cells)")
    ax2 = fig.add_axes([0.71, 0.18, 0.27, 0.68])
    st_axis = freqs * to_st
    ax2.semilogy(st_axis, spec / spec.max(), "k-", lw=1)
    ax2.axvline(st_probe, color="C3", ls="--", lw=1, label=f"probe FFT: St = {st_probe:.4f}")
    ax2.set(xlim=(0, 0.8), ylim=(1e-5, 2), xlabel="St = f D / U", title="Cross-stream velocity spectrum")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.savefig("docs/wake.png", dpi=120)
    plt.close(fig)

    # ---- figure 2: POD
    fig = plt.figure(figsize=(12, 6))
    a = fig.add_axes([0.06, 0.58, 0.25, 0.36])
    a.semilogy(np.arange(1, 41), p.sigma[:40] / p.sigma[0], "o-", ms=4)
    a.set(xlabel="mode", ylabel="σ_k / σ_1", title="Singular values (pairs = travelling waves)")
    a.grid(alpha=0.3)
    b = fig.add_axes([0.06, 0.1, 0.25, 0.36])
    b.plot(np.arange(1, 41), 100 * cum[:40], "o-", ms=4)
    b.set(xlabel="modes kept", ylabel="cumulative energy [%]", ylim=(40, 101))
    b.grid(alpha=0.3)
    for k in range(6):
        ax = fig.add_axes([0.36 + (k % 3) * 0.215, 0.53 - (k // 3) * 0.43, 0.2, 0.4])
        mode = p.modes[:, k].reshape(nx, ny)
        show(ax, mode, data, np.abs(mode).max())
        ax.set_title(f"POD mode {k + 1}  ({100 * p.energy[k]:.1f} %)", fontsize=9)
    fig.savefig("docs/pod.png", dpi=110)
    plt.close(fig)

    # ---- figure 3: DMD spectrum and modes
    fig = plt.figure(figsize=(12, 6.2))
    a = fig.add_axes([0.05, 0.55, 0.24, 0.4])
    th = np.linspace(0, 2 * np.pi, 400)
    a.plot(np.cos(th), np.sin(th), "k-", lw=0.6)
    a.scatter(d.eigs.real, d.eigs.imag, c=np.log10(np.abs(d.amplitudes)), cmap="viridis", s=30, zorder=3)
    a.set(aspect="equal", xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), title="DMD eigenvalues λ (unit circle)")
    a.grid(alpha=0.3)
    b = fig.add_axes([0.05, 0.08, 0.24, 0.35])
    b.stem(st_dmd[pos], np.abs(d.amplitudes[pos]) / np.abs(d.amplitudes[pos]).max(), basefmt=" ")
    b.set(xlabel="St", ylabel="|b| (normalised)", title="DMD amplitude spectrum")
    b.grid(alpha=0.3)
    harmonics = [k for k in order if d.frequency[k] > 1e-12][:3]
    harmonics = sorted(harmonics, key=lambda k: d.frequency[k])
    for j, k in enumerate(harmonics):
        mode = d.modes[:, k].reshape(nx, ny)
        for i, (part, name) in enumerate(((mode.real, "Re"), (mode.imag, "Im"))):
            ax = fig.add_axes([0.34 + j * 0.22, 0.53 - i * 0.45, 0.21, 0.4])
            show(ax, part, data, np.abs(mode).max())
            ax.set_title(f"{name} φ, St = {st_dmd[k]:.4f}", fontsize=9)
    fig.savefig("docs/dmd.png", dpi=110)
    plt.close(fig)

    # ---- figure 4: reconstruction error
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.semilogy(ranks, e_pod, "o-", label="POD (optimal, Eckart–Young)")
    ax.semilogy(ranks, e_dmd, "s--", label="DMD (+ mean mode)")
    ax.set(xlabel="rank", ylabel="relative error (fluctuations)", title="Low-rank reconstruction")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig("docs/reconstruction.png", dpi=120)
    plt.close(fig)

    # ---- GIF: full field vs rank-6 POD reconstruction
    rec = p.reconstruct(6).T.reshape(nt, nx, ny)
    frames = range(0, min(nt, 240), 2)
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.2))
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.93, hspace=0.18)
    ims = [show(axes[0], vort[0], data, vmax), show(axes[1], rec[0], data, vmax)]
    axes[0].set_title("Lattice-Boltzmann vorticity", fontsize=9)
    axes[1].set_title("Rank-6 POD reconstruction", fontsize=9)

    def update(k):
        ims[0].set_data(vort[k].T)
        ims[1].set_data(rec[k].T)
        return ims

    FuncAnimation(fig, update, frames=frames).save("docs/wake.gif", writer=PillowWriter(fps=20), dpi=65)
    plt.close(fig)


if __name__ == "__main__":
    main()
