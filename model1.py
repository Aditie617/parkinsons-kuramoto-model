import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Set random seed for reproducibility
np.random.seed(42)

class ParkinsonKuramotoPipeline:
    def __init__(self, N=200, T=4000, dt=0.002):
        self.N = N          # Number of oscillators
        self.T = T          # Total time steps (Extended)
        self.dt = dt        # Step size (Smaller for high-freq stability)
        self.time = np.arange(0, T * dt, dt)
        
        # Intrinsic frequencies: Centered in the Beta Band (~20 Hz)
        self.omega = np.random.normal(loc=20.0, scale=2.0, size=N)
        # Initialize phases randomly between [0, 2pi)
        self.theta = np.random.uniform(0, 2 * np.pi, size=N)
        
        # Optimized Dopamine & Coupling parameters for distinct bifurcation
        self.da_0 = 0.5     # Threshold inflection point
        self.alpha = 12.0   # Sharper sigmoidal transition
        self.K_min = 0.1    # Healthy weak coupling (low synchrony)
        self.K_max = 15.0   # Pathological strong coupling (high synchrony)
        self.gamma = 0.3    # Calibrated decay rate
        
    def simulate(self):
        # Initialize tracking arrays
        dopamine_hist = np.zeros(self.T)
        K_hist = np.zeros(self.T)
        R_hist = np.zeros(self.T)
        phase_hist = np.zeros((self.T, self.N))
        
        # Initial dopamine concentration
        da = 1.0
        
        for t in range(self.T):
            # 1. Update Dopamine (Decay)
            da += -self.gamma * da * self.dt
            dopamine_hist[t] = da
            
            # 2. Compute Adaptive Coupling K(t)
            K = self.K_min + (self.K_max - self.K_min) / (1 + np.exp(-self.alpha * (self.da_0 - da)))
            K_hist[t] = K
            
            # 3. Calculate Kuramoto Order Parameter R(t)
            order_param = np.mean(np.exp(1j * self.theta))
            R_hist[t] = np.abs(order_param)
            phase_hist[t, :] = self.theta
            
            # 4. Compute Phase Interdependence (Vectorized)
            phase_diffs = np.sin(np.subtract.outer(self.theta, self.theta))
            dtheta_dt = self.omega + (K / self.N) * np.sum(phase_diffs, axis=1)
            
            # 5. Integrate Phase
            self.theta = (self.theta + dtheta_dt * self.dt) % (2 * np.pi)
            
        return dopamine_hist, K_hist, R_hist, phase_hist

    def plot_publication_figures(self, da_hist, K_hist, R_hist, phase_hist):
        # Set academic plot styling
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'figure.titlesize': 16
        })
        
        # --- FIGURE 1: Dynamic Phase Transition ---
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        color = '#1f77b4'
        ax1.set_xlabel('Simulated Time (s)', fontweight='bold')
        ax1.set_ylabel('Dopamine [DA] & Coupling Strength [K]', color=color, fontweight='bold')
        line1 = ax1.plot(self.time, da_hist, label='Dopamine $[DA]$', color=color, linestyle='--', linewidth=2)
        line2 = ax1.plot(self.time, K_hist, label='Coupling Strength $[K]$', color='g', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        ax2 = ax1.twinx()  
        color = '#d62728'
        ax2.set_ylabel('Phase Synchrony Index ($R$)', color=color, fontweight='bold')
        line3 = ax2.plot(self.time, R_hist, label='Order Parameter $R(t)$', color=color, linewidth=2.5)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Find and mark bifurcation point (Onset)
        bifurcation_indices = np.where(R_hist > 0.6)[0]
        if len(bifurcation_indices) > 0:
            onset_time = self.time[bifurcation_indices[0]]
            ax1.axvline(x=onset_time, color='purple', linestyle=':', linewidth=2, label='Predicted Tremor Onset')
            
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        plt.title('Dopamine-Dependent Bifurcation and Phase Locking', pad=15)
        plt.tight_layout()
        plt.savefig('bifurcation_plot.png', dpi=300, bbox_inches='tight')
        plt.show()

        # --- FIGURE 2: Spectral Analysis (FFT of Collective Signal) ---
        lfp = np.mean(np.sin(phase_hist), axis=1)
        half = self.T // 2
        lfp_healthy = lfp[:half]
        lfp_patho = lfp[half:]
        
        yf_h = fft(lfp_healthy)
        xf_h = fftfreq(half, self.dt)[:half//2]
        yf_p = fft(lfp_patho)
        xf_p = fftfreq(self.T - half, self.dt)[:(self.T - half)//2]
        
        plt.figure(figsize=(8, 4))
        plt.plot(xf_h, 2.0/half * np.abs(yf_h[0:half//2]), label='Early Phase (High Entropy)', color='gray', alpha=0.7)
        plt.plot(xf_p, 2.0/(self.T-half) * np.abs(yf_p[0:(self.T-half)//2]), label='Late Phase (Pathological Beta Sync)', color='red', linewidth=2)
        plt.xlim(0, 40)
        plt.xlabel('Frequency (Hz)', fontweight='bold')
        plt.ylabel('Spectral Amplitude', fontweight='bold')
        plt.title('Power Spectrum Shift: Broadband Decay to Narrowband Beta Peak')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig('spectral_analysis.png', dpi=300)
        plt.show()

# Execute Pipeline
pipeline = ParkinsonKuramotoPipeline()
da, K, R, phases = pipeline.simulate()
pipeline.plot_publication_figures(da, K, R, phases)