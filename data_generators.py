"""
Time series generators for exploring attention map behavior in transformers.
Each generator produces time series with distinct distributional characteristics.
"""

import numpy as np
from scipy import stats
from scipy.signal import lfilter


class TimeSeriesGenerator:
    """Base class for time series generators."""
    
    def __init__(self, length=1000, seed=None):
        self.length = length
        self.rng = np.random.RandomState(seed)
    
    def generate(self):
        """Generate time series. Must be implemented by subclasses."""
        raise NotImplementedError


class HeavyTailedAR1(TimeSeriesGenerator):
    """
    AR(1) process with heavy-tailed noise (Student-t instead of Gaussian).
    Effect: Most changes are small, occasionally enormous spikes.
    """
    def __init__(self, phi=0.7, df=3.0, scale=1.0, length=1000, seed=None):
        super().__init__(length, seed)
        self.phi = phi  # AR coefficient
        self.df = df  # Degrees of freedom for Student-t (lower = heavier tails)
        self.scale = scale
    
    def generate(self):
        series = np.zeros(self.length)
        for t in range(1, self.length):
            noise = stats.t.rvs(df=self.df, scale=self.scale, random_state=self.rng)
            series[t] = self.phi * series[t-1] + noise
        return series


class GARCH11(TimeSeriesGenerator):
    """
    GARCH(1,1) process: Time-varying volatility with clustering.
    Effect: Quiet periods followed by violent bursts of volatility.
    """
    def __init__(self, omega=0.1, alpha=0.1, beta=0.8, length=1000, seed=None):
        super().__init__(length, seed)
        self.omega = omega  # Long-term variance
        self.alpha = alpha  # ARCH coefficient
        self.beta = beta  # GARCH coefficient
    
    def generate(self):
        returns = np.zeros(self.length)
        variance = np.zeros(self.length)
        variance[0] = self.omega / (1 - self.alpha - self.beta)
        
        for t in range(1, self.length):
            variance[t] = self.omega + self.alpha * returns[t-1]**2 + self.beta * variance[t-1]
            returns[t] = self.rng.normal(0, np.sqrt(variance[t]))
        
        # Convert to cumulative series
        series = np.cumsum(returns)
        return series


class JumpDiffusion(TimeSeriesGenerator):
    """
    Jump diffusion: Drift-diffusion process with random massive jumps.
    Effect: Smooth movement with occasional large discontinuous jumps.
    """
    def __init__(self, mu=0.0, sigma=1.0, jump_prob=0.05, jump_size=5.0, length=1000, seed=None):
        super().__init__(length, seed)
        self.mu = mu  # Drift
        self.sigma = sigma  # Diffusion volatility
        self.jump_prob = jump_prob  # Probability of jump per timestep
        self.jump_size = jump_size  # Size of jumps
    
    def generate(self):
        series = np.zeros(self.length)
        for t in range(1, self.length):
            # Regular diffusion
            diffusion = self.rng.normal(self.mu, self.sigma)
            
            # Random jump
            if self.rng.random() < self.jump_prob:
                jump = self.rng.normal(0, self.jump_size)
            else:
                jump = 0
            
            series[t] = series[t-1] + diffusion + jump
        return series


class RegimeSwitching(TimeSeriesGenerator):
    """
    Regime-switching: Mean shifts abruptly based on hidden Markov state.
    Effect: Abrupt changes in distributional properties.
    """
    def __init__(self, means=[-2.0, 2.0], volatilities=[0.5, 2.0], 
                 transition_probs=[[0.95, 0.05], [0.05, 0.95]], length=1000, seed=None):
        super().__init__(length, seed)
        self.means = means
        self.volatilities = volatilities
        self.transition_probs = np.array(transition_probs)
        self.n_regimes = len(means)
    
    def generate(self):
        series = np.zeros(self.length)
        state = self.rng.randint(0, self.n_regimes)
        
        for t in range(self.length):
            # Generate observation from current state
            series[t] = self.rng.normal(self.means[state], self.volatilities[state])
            
            # Transition to next state
            state = self.rng.choice(self.n_regimes, p=self.transition_probs[state])
        
        return series


class TrendBreaks(TimeSeriesGenerator):
    """
    Trend breaks: Linear trend that abruptly changes slope.
    Effect: Structural breaks in the data generating process.
    """
    def __init__(self, slopes=[0.1, -0.1, 0.05], break_points=[300, 700], 
                 noise_std=0.5, length=1000, seed=None):
        super().__init__(length, seed)
        self.slopes = slopes
        self.break_points = break_points
        self.noise_std = noise_std
    
    def generate(self):
        series = np.zeros(self.length)
        current_slope_idx = 0
        
        for t in range(self.length):
            if current_slope_idx < len(self.break_points) and t >= self.break_points[current_slope_idx]:
                current_slope_idx += 1
            
            slope = self.slopes[min(current_slope_idx, len(self.slopes) - 1)]
            series[t] = series[t-1] + slope + self.rng.normal(0, self.noise_std)
        
        return series


class RandomWalk(TimeSeriesGenerator):
    """
    Random walk: X_t = X_{t-1} + epsilon (no mean reversion).
    Effect: Non-stationary process with no tendency to revert.
    """
    def __init__(self, mu=0.0, sigma=1.0, length=1000, seed=None):
        super().__init__(length, seed)
        self.mu = mu
        self.sigma = sigma
    
    def generate(self):
        innovations = self.rng.normal(self.mu, self.sigma, self.length)
        series = np.cumsum(innovations)
        return series


class SeasonTrendOutliers(TimeSeriesGenerator):
    """
    Seasonal + Trend + Outliers: Sinusoidal wave with linear trend and sparse outliers.
    Effect: Predictable pattern with occasional disruptions.
    """
    def __init__(self, trend=0.01, amplitude=2.0, frequency=0.1, 
                 outlier_prob=0.02, outlier_size=5.0, noise_std=0.3, length=1000, seed=None):
        super().__init__(length, seed)
        self.trend = trend
        self.amplitude = amplitude
        self.frequency = frequency
        self.outlier_prob = outlier_prob
        self.outlier_size = outlier_size
        self.noise_std = noise_std
    
    def generate(self):
        t = np.arange(self.length)
        trend_component = self.trend * t
        seasonal_component = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        noise = self.rng.normal(0, self.noise_std, self.length)
        
        # Add outliers
        outliers = np.zeros(self.length)
        outlier_mask = self.rng.random(self.length) < self.outlier_prob
        outliers[outlier_mask] = self.rng.normal(0, self.outlier_size, np.sum(outlier_mask))
        
        series = trend_component + seasonal_component + noise + outliers
        return series


class MultiSeasonality(TimeSeriesGenerator):
    """
    Multi-seasonality: Complex overlapping cycles (e.g., daily + weekly patterns).
    Effect: Multiple predictable patterns at different frequencies.
    """
    def __init__(self, amplitudes=[3.0, 1.5], frequencies=[0.1, 0.05], 
                 phases=[0, np.pi/4], noise_std=0.2, length=1000, seed=None):
        super().__init__(length, seed)
        self.amplitudes = amplitudes
        self.frequencies = frequencies
        self.phases = phases
        self.noise_std = noise_std
    
    def generate(self):
        t = np.arange(self.length)
        series = np.zeros(self.length)
        
        for amp, freq, phase in zip(self.amplitudes, self.frequencies, self.phases):
            series += amp * np.sin(2 * np.pi * freq * t + phase)
        
        series += self.rng.normal(0, self.noise_std, self.length)
        return series


class OneOverFNoise(TimeSeriesGenerator):
    """
    1/f Noise (Pink Noise): Process where correlations decay very slowly.
    Effect: Long memory with persistent correlations across time.
    """
    def __init__(self, length=1000, seed=None):
        super().__init__(length, seed)
        # For 1/f noise, we use a filter approximation
        # Generate white noise and filter it
    
    def generate(self):
        # Generate white noise
        white_noise = self.rng.normal(0, 1, self.length * 2)
        
        # Create 1/f filter (simplified approximation)
        # Using a filter that approximates 1/f behavior
        # This is a simplified approach - true 1/f noise requires more sophisticated methods
        b = [1]
        a = [1, -0.5]  # Simple filter approximation
        
        filtered = lfilter(b, a, white_noise)
        series = filtered[:self.length]
        
        # Normalize
        series = (series - np.mean(series)) / np.std(series)
        
        return series


def get_all_generators(length=1000, seed=42):
    """Get all time series generators with default parameters."""
    generators = {
        'HeavyTailedAR1': HeavyTailedAR1(length=length, seed=seed),
        'GARCH11': GARCH11(length=length, seed=seed),
        'JumpDiffusion': JumpDiffusion(length=length, seed=seed),
        'RegimeSwitching': RegimeSwitching(length=length, seed=seed),
        'TrendBreaks': TrendBreaks(length=length, seed=seed),
        'RandomWalk': RandomWalk(length=length, seed=seed),
        'SeasonTrendOutliers': SeasonTrendOutliers(length=length, seed=seed),
        'MultiSeasonality': MultiSeasonality(length=length, seed=seed),
        'OneOverFNoise': OneOverFNoise(length=length, seed=seed),
    }
    return generators

