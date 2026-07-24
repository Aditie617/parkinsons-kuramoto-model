import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.fft import fft, fftfreq

np.random.seed(42)


class AdvancedParkinsonNetwork:
    def __init__(self, N=100, T=4000, dt=0.002):
        self.N = N
        self.T = T
        self.dt = dt
        self.time = np.arange(0, T * dt, dt)

        # 1. Intrinsic frequencies, drawn in Hz then converted to rad/s.
        #    This is the fix: the phase ODE dtheta/dt = omega + ... needs
        #    omega in rad/s, not Hz. Feeding it Hz directly (the original
        #    bug) silently divides the true oscillation frequency by 2*pi,
        #    which is why the spectra were landing around 3 Hz instead of
        #    the intended beta band.
        mu_hz, sigma_hz = 28.0, 1.2
        self.omega = 2 * np.pi * np.random.normal(loc=mu_hz, scale=sigma_hz, size=N)

        # 2. Watts-Strogatz small-world topology
        G = nx.watts_strogatz_graph(n=N, k=10, p=0.1, seed=42)
        self.A = nx.to_numpy_array(G)
        self.degrees = np.sum(self.A, axis=1)
        self.degrees[self.degrees == 0] = 1

        # 3. Axonal conduction delay (mean ~10 ms at dt = 0.002 s -> 5 steps)
        self.mean_delay_steps = 3
        self.delays = np.random.poisson(lam=self.mean_delay_steps, size=(N, N))
        self.max_delay = int(np.max(self.delays)) + 1
        self.cols = np.tile(np.arange(N), (N, 1))  # column index grid for advanced indexing

        self.phase_buffer = np.zeros((self.max_delay, N))
        self.phase_buffer[:] = np.random.uniform(0, 2 * np.pi, size=(self.max_delay, N))

        # 4. Noise intensity D (variance-rate of the phase Wiener process)
        self.D = 0.6

        # 5. Dopamine-adaptive coupling parameters (match paper Eq. 2-3)
        self.da_0 = 0.5
        self.alpha = 10.0
        self.K_min = 1.5
        self.K_max = 130.0
        self.gamma = 0.4

    def simulate(self):
        da_hist = np.zeros(self.T)
        K_hist = np.zeros(self.T)
        R_hist = np.zeros(self.T)
        lfp_hist = np.zeros(self.T)

        da = 1.0
        current_phases = self.phase_buffer[-1, :].copy()
        buf_idx = self.max_delay - 1

        for t in range(self.T):
            da += -self.gamma * da * self.dt
            da_hist[t] = da

            K = self.K_min + (self.K_max - self.K_min) / (1 + np.exp(-self.alpha * (self.da_0 - da)))
            K_hist[t] = K

            R_hist[t] = np.abs(np.mean(np.exp(1j * current_phases)))
            lfp_hist[t] = np.mean(np.sin(current_phases))

            # Vectorized delayed coupling: for every edge (i, j) pull the
            # phase of oscillator j at the delay appropriate to that edge.
            delayed_idx = (buf_idx - self.delays) % self.max_delay          # (N, N)
            phase_j_delayed = self.phase_buffer[delayed_idx, self.cols]      # (N, N)
            coupling = np.sum(self.A * np.sin(phase_j_delayed - current_phases[:, None]), axis=1)

            # Correct Euler-Maruyama noise: dW ~ N(0, dt), added directly
            # to the phase increment (not pre-divided by dt then re-multiplied).
            drift = self.omega + (K / self.degrees) * coupling
            dW = np.random.normal(0, np.sqrt(self.dt), size=self.N)

            current_phases = (current_phases + drift * self.dt + np.sqrt(2 * self.D) * dW) % (2 * np.pi)
            buf_idx = (buf_idx + 1) % self.max_delay
            self.phase_buffer[buf_idx, :] = current_phases

        return da_hist, K_hist, R_hist, lfp_hist

    def compute_metrics(self, lfp):
        half = self.T // 2

        def window_entropy(signal):
            yf = np.abs(fft(signal))[: len(signal) // 2]
            psd = yf ** 2
            psd_norm = psd / np.sum(psd)
            return -np.sum(psd_norm * np.log(psd_norm + 1e-12)) / np.log(len(psd_norm))

        return window_entropy(lfp[:half]), window_entropy(lfp[half:])

    def plot_publication_analytics(self, da_hist, K_hist, R_hist, lfp):
        se_h, se_p = self.compute_metrics(lfp)

        plt.rcParams.update({
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.labelsize": 12.5,
            "axes.titlesize": 12.5,
            "axes.linewidth": 0.9,
        })
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

        # ---- Panel A: phase transition dynamics ----
        ax1 = axes[0]
        c1 = "#1f4e8c"
        ax1.set_xlabel("Simulated time (s)")
        ax1.set_ylabel("Dopamine [DA] / Coupling K", color=c1)
        ax1.plot(self.time, da_hist, label="Dopamine [DA]", color=c1, ls="--", lw=1.6)
        ax1.plot(self.time, K_hist, label="Coupling K", color="#2ca02c", lw=1.8)
        ax1.tick_params(axis="y", labelcolor=c1)
        ax1.grid(True, ls=":", alpha=0.4)
        ax1.spines["top"].set_visible(False)

        ax2 = ax1.twinx()
        c2 = "#8c1f1f"
        ax2.set_ylabel("Network synchrony index (R)", color=c2)
        ax2.plot(self.time, R_hist, label="Order parameter R(t)", color=c2, lw=1.9)
        ax2.tick_params(axis="y", labelcolor=c2)
        ax2.set_ylim(-0.02, 1.02)
        ax2.spines["top"].set_visible(False)

        bifurcation_idx = np.where(R_hist > 0.6)[0]
        onset = None
        if len(bifurcation_idx) > 0:
            onset = self.time[bifurcation_idx[0]]
            ax1.axvline(onset, color="#7a3fa0", ls=":", lw=1.6)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right", frameon=False)
        ax1.set_title("A: Phase transition dynamics with noise and delay" +
                      (f"  (onset {onset:.2f} s)" if onset else ""))

        # ---- Panel B: spectral shift ----
        ax3 = axes[1]
        half = self.T // 2
        yf_h = fft(lfp[:half])
        xf_h = fftfreq(half, self.dt)[: half // 2]
        yf_p = fft(lfp[half:])
        xf_p = fftfreq(self.T - half, self.dt)[: (self.T - half) // 2]

        ax3.axvspan(13, 30, color="#f4d78a", alpha=0.5, label="Human beta band (13-30 Hz)")
        ax3.plot(xf_h, 2.0 / half * np.abs(yf_h[: half // 2]),
                 label=f"Healthy state (Hsp = {se_h:.2f})", color="#9a9a9a", lw=1.2)
        ax3.plot(xf_p, 2.0 / (self.T - half) * np.abs(yf_p[: (self.T - half) // 2]),
                 label=f"Pathological state (Hsp = {se_p:.2f})", color="#d62728", lw=1.7)
        ax3.set_xlim(0, 45)
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Spectral magnitude")
        ax3.legend(fontsize=9, frameon=False, loc="upper right")
        ax3.grid(True, ls=":", alpha=0.4)
        ax3.spines["top"].set_visible(False)
        ax3.set_title("B: Spectral entropy collapse into the beta band")

        plt.tight_layout(pad=1.0)
        plt.savefig("A Dopamine-Modulated Adaptive Kuramoto Network for Modeling Parkinsonian Tremor Emergence.png",
                     dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Final R: {R_hist[-1]:.3f} | Onset: {onset} | Hsp healthy: {se_h:.3f} | Hsp patho: {se_p:.3f}")
        peak_h = xf_h[np.argmax(np.abs(yf_h[: half // 2]))]
        peak_p = xf_p[np.argmax(np.abs(yf_p[: (self.T - half) // 2]))]
        print(f"Peak freq healthy: {peak_h:.2f} Hz | Peak freq pathological: {peak_p:.2f} Hz")


network = AdvancedParkinsonNetwork()
da, K, R, lfp = network.simulate()
network.plot_publication_analytics(da, K, R, lfp)
