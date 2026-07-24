import numpy as np
import matplotlib.pyplot as plt
import networkx as nx 
from scipy.fft import fft, fftfreq
from scipy.stats import entropy

# Set random seed for exact reproducibility
np.random.seed(42)

class AdvancedParkinsonNetwork:
    def __init__(self, N=100, T=3000, dt=0.002):
        self.N = N
        self.T = T
        self.dt = dt
        self.time = np.arange(0, T * dt, dt)
        
        # 1. Intrinsic Frequencies (Beta Band: ~20 Hz)
        self.omega = np.random.normal(loc=20.0, scale=1.5, size=N)
        
        # 2. Network Topology: Watts-Strogatz Small-World Graph
        # Mimics biological brain connectivity (high clustering, short path lengths)
        G = nx.watts_strogatz_graph(n=N, k=10, p=0.1, seed=42)
        self.A = nx.to_numpy_array(G)
        self.degrees = np.sum(self.A, axis=1)
        self.degrees[self.degrees == 0] = 1 # Avoid division by zero
        
        # 3. Time Delays (Conduction latency in steps)
        # Mean delay approx 10ms (5 time steps given dt=0.002s)
        self.mean_delay_steps = 5
        self.delays = np.random.poisson(lam=self.mean_delay_steps, size=(N, N))
        self.max_delay = int(np.max(self.delays)) + 1
        
        # Initialize Phase History Buffer to handle time delays
        self.phase_buffer = np.zeros((self.max_delay, N))
        self.phase_buffer[:] = np.random.uniform(0, 2 * np.pi, size=(self.max_delay, N))
        
        # 4. Stochastic Noise Intensity
        self.noise_intensity = 1.5
        
        # 5. Dopamine-Adaptive System Parameters
        self.da_0 = 0.5
        self.alpha = 10.0
        self.K_min = 0.2
        self.K_max = 25.0
        self.gamma = 0.4  # Dopamine decay rate

    def simulate(self):
        da_hist = np.zeros(self.T)
        K_hist = np.zeros(self.T)
        R_hist = np.zeros(self.T)
        lfp_hist = np.zeros(self.T)
        spectral_entropy_hist = np.zeros(self.T // 100) # Checked in windows
        
        da = 1.0
        current_phases = self.phase_buffer[-1, :].copy()
        
        # Buffer pointer setup
        buf_idx = self.max_delay - 1
        
        for t in range(self.T):
            # Update Dopamine & Coupling Strength
            da += -self.gamma * da * self.dt
            da_hist[t] = da
            K = self.K_min + (self.K_max - self.K_min) / (1 + np.exp(-self.alpha * (self.da_0 - da)))
            K_hist[t] = K
            
            # Global Order Parameter R(t)
            R_hist[t] = np.abs(np.mean(np.exp(1j * current_phases)))
            lfp_hist[t] = np.mean(np.sin(current_phases))
            
            # Compute Phase Interdependence with Time Delays
            dtheta_dt = np.zeros(self.N)
            for i in range(self.N):
                coupling_sum = 0
                for j in range(self.N):
                    if self.A[i, j] > 0:
                        # Extract delayed phase from history buffer
                        delay_step = self.delays[i, j]
                        delayed_idx = (buf_idx - delay_step) % self.max_delay
                        phase_j_delayed = self.phase_buffer[delayed_idx, j]
                        
                        coupling_sum += self.A[i, j] * np.sin(phase_j_delayed - current_phases[i])
                        
                # Euler-Maruyama: Deterministic steps + Stochastic Noise
                noise = np.random.normal(0, np.sqrt(self.dt)) * self.noise_intensity
                dtheta_dt[i] = self.omega[i] + (K / self.degrees[i]) * coupling_sum + noise
            
            # Update current phases and advance history ring buffer
            current_phases = (current_phases + dtheta_dt * self.dt) % (2 * np.pi)
            buf_idx = (buf_idx + 1) % self.max_delay
            self.phase_buffer[buf_idx, :] = current_phases
            
        return da_hist, K_hist, R_hist, lfp_hist

    def compute_metrics(self, lfp):
        # Calculate Spectral Entropy for Early (Healthy) vs Late (Pathological)
        half = self.T // 2
        
        def calculate_window_entropy(signal):
            yf = np.abs(fft(signal))[:len(signal)//2]
            psd = yf**2
            psd_norm = psd / np.sum(psd)
            return -np.sum(psd_norm * np.log(psd_norm + 1e-12)) / np.log(len(psd_norm))
            
        se_healthy = calculate_window_entropy(lfp[:half])
        se_patho = calculate_window_entropy(lfp[half:])
        return se_healthy, se_patho

    def plot_publication_analytics(self, da_hist, K_hist, R_hist, lfp):
        se_h, se_p = self.compute_metrics(lfp)
        
        plt.rcParams.update({'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13})
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # LEFT PANEL: Bifurcation Under Constraints
        ax1 = axes[0]
        color = '#1f77b4'
        ax1.set_xlabel('Simulated Time (s)', fontweight='bold')
        ax1.set_ylabel('Dopamine [DA] / Coupling [K]', color=color)
        ax1.plot(self.time, da_hist, label='Dopamine $[DA]$', color=color, linestyle='--')
        ax1.plot(self.time, K_hist, label='Coupling Strength $[K]$', color='g')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle=':', alpha=0.5)
        
        ax2 = ax1.twinx()
        color = '#d62728'
        ax2.set_ylabel('Network Synchrony Index ($R$)', color=color, fontweight='bold')
        ax2.plot(self.time, R_hist, label='Order Parameter $R(t)$', color=color, alpha=0.8)
        ax2.tick_params(axis='y', labelcolor=color)
        ax1.set_title("A: Phase Transition Dynamics with Noise & Delays")
        
        # RIGHT PANEL: Power Spectral Shift vs. Electrophysiology Literature
        ax3 = axes[1]
        half = self.T // 2
        yf_h = fft(lfp[:half])
        xf_h = fftfreq(half, self.dt)[:half//2]
        yf_p = fft(lfp[half:])
        xf_p = fftfreq(self.T - half, self.dt)[:(self.T - half)//2]
        
     # Change lines 140-141 to include the index slicing [:half//2] on the Y arrays:
        ax3.plot(xf_h, 2.0/half * np.abs(yf_h[:half//2]), label=f'Healthy State ($H_{{sp}}$: {se_h:.2f})', color='gray', alpha=0.7)
        ax3.plot(xf_p, 2.0/(self.T-half) * np.abs(yf_p[:(self.T-half)//2]), label=f'Pathological State ($H_{{sp}}$: {se_p:.2f})', color='red', linewidth=2)
        # Overlay experimental anchor data points typical in literature (Brown et al., 2001)
        ax3.axvspan(13, 30, color='orange', alpha=0.15, label="Lit. Human Beta Oscillation Band")
        ax3.set_xlim(0, 45)
        ax3.set_xlabel('Frequency (Hz)', fontweight='bold')
        ax3.set_ylabel('Spectral Magnitude', fontweight='bold')
        ax3.set_title("B: Spectral Entropy Decay & Electrophysiology Fit")
        ax3.legend()
        ax3.grid(True, linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('advanced_parkinson_analytics.png', dpi=300)
        plt.show()

# Run the complete advanced pipeline
network = AdvancedParkinsonNetwork()
da, K, R, lfp = network.simulate()
network.plot_publication_analytics(da, K, R, lfp)