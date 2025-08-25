"""
Biological Environment for Reinforcement Learning

This module implements the environment interface for training RL agents
to optimize synthetic biological oscillators.
"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional, Any, Union
from ..edelstein_oscillator.oscillator import EdelsteinOscillator
from ..otero_repressilator.repressilator import OteroRepressilator


class BioOscillatorEnvironment(gym.Env):
    """
    OpenAI Gym Environment for Biological Oscillator Optimization
    
    This environment allows RL agents to learn optimal parameters
    for synthetic biological oscillators through trial and error.
    """
    
    def __init__(self,
                 oscillator_type: str = "edelstein",
                 target_period: float = 10.0,
                 target_amplitude: float = 2.0,
                 simulation_time: float = 50.0,
                 noise_strength: float = 0.1,
                 reward_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the biological oscillator environment.
        
        Args:
            oscillator_type: Type of oscillator ("edelstein" or "otero")
            target_period: Target oscillation period
            target_amplitude: Target oscillation amplitude
            simulation_time: Time span for each simulation
            noise_strength: Biological noise level
            reward_weights: Weights for different reward components
        """
        super(BioOscillatorEnvironment, self).__init__()
        
        self.oscillator_type = oscillator_type.lower()
        self.target_period = target_period
        self.target_amplitude = target_amplitude
        self.simulation_time = simulation_time
        self.noise_strength = noise_strength
        
        # Default reward weights
        self.reward_weights = reward_weights or {
            'period': 0.4,
            'amplitude': 0.3,
            'stability': 0.2,
            'oscillation': 0.1
        }
        
        # Define parameter bounds and action/observation spaces
        self._setup_spaces()
        
        # Environment state
        self.current_parameters = None
        self.step_count = 0
        self.max_steps = 100
        self.best_reward = -np.inf
        self.episode_rewards = []
        
        # Metrics tracking
        self.metrics_history = []
        self.parameter_history = []
        
    def _setup_spaces(self):
        """Setup action and observation spaces based on oscillator type."""
        
        if self.oscillator_type == "edelstein":
            # Edelstein oscillator parameters
            self.param_bounds = {
                'k1': (0.1, 5.0),
                'k2': (0.1, 5.0),
                'k3': (0.1, 5.0),
                'k4': (0.1, 5.0),
                'K1': (0.1, 5.0),
                'K2': (0.1, 5.0)
            }
            
        elif self.oscillator_type == "otero":
            # Otero repressilator parameters
            self.param_bounds = {
                'alpha': (50.0, 500.0),
                'alpha0': (0.0, 10.0),
                'beta': (0.1, 1.0),
                'gamma': (0.5, 5.0),
                'n': (1.0, 4.0)
            }
        else:
            raise ValueError(f"Unknown oscillator type: {self.oscillator_type}")
        
        self.param_names = list(self.param_bounds.keys())
        n_params = len(self.param_names)
        
        # Action space: continuous parameter adjustments (normalized to [-1, 1])
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_params,), dtype=np.float32
        )
        
        # Observation space: current parameters + performance metrics + targets
        obs_dim = n_params + 6  # params + [period, amplitude, stability, oscillating, target_period, target_amplitude]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        # Initialize random parameters within bounds
        self.current_parameters = {}
        for param, (low, high) in self.param_bounds.items():
            self.current_parameters[param] = np.random.uniform(low, high)
        
        self.step_count = 0
        self.episode_rewards = []
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Parameter adjustments (normalized to [-1, 1])
            
        Returns:
            observation, reward, done, info
        """
        self.step_count += 1
        
        # Apply action to update parameters
        self._apply_action(action)
        
        # Simulate oscillator with current parameters
        metrics = self._simulate_oscillator()
        
        # Calculate reward
        reward = self._calculate_reward(metrics)
        
        # Track metrics
        self.episode_rewards.append(reward)
        self.metrics_history.append(metrics)
        self.parameter_history.append(self.current_parameters.copy())
        
        # Check if episode is done
        done = (self.step_count >= self.max_steps) or self._is_goal_reached(metrics)
        
        # Update best reward
        if reward > self.best_reward:
            self.best_reward = reward
        
        # Prepare info dictionary
        info = {
            'metrics': metrics,
            'parameters': self.current_parameters.copy(),
            'step': self.step_count,
            'best_reward': self.best_reward,
            'goal_reached': self._is_goal_reached(metrics)
        }
        
        return self._get_observation(), reward, done, info
    
    def _apply_action(self, action: np.ndarray):
        """Apply action to update current parameters."""
        for i, param in enumerate(self.param_names):
            low, high = self.param_bounds[param]
            current_val = self.current_parameters[param]
            
            # Scale action from [-1, 1] to parameter range
            adjustment_range = 0.1 * (high - low)  # 10% of parameter range
            adjustment = action[i] * adjustment_range
            
            # Apply adjustment with bounds checking
            new_val = current_val + adjustment
            self.current_parameters[param] = np.clip(new_val, low, high)
    
    def _simulate_oscillator(self) -> Dict:
        """Simulate the oscillator with current parameters."""
        try:
            if self.oscillator_type == "edelstein":
                oscillator = EdelsteinOscillator(
                    noise_strength=self.noise_strength,
                    **self.current_parameters
                )
            else:  # otero
                oscillator = OteroRepressilator(
                    noise_strength=self.noise_strength,
                    **self.current_parameters
                )
            
            # Run simulation
            oscillator.simulate(
                time_span=(0, self.simulation_time),
                num_points=int(self.simulation_time * 20)  # 20 points per time unit
            )
            
            # Get fitness metrics
            metrics = oscillator.get_fitness_metrics()
            
            return metrics
            
        except Exception as e:
            # Return failure metrics if simulation fails
            return {
                'period': 0,
                'amplitude': 0,
                'stability': 0,
                'is_oscillating': False,
                'simulation_failed': True
            }
    
    def _calculate_reward(self, metrics: Dict) -> float:
        """
        Calculate reward based on oscillator performance.
        
        Args:
            metrics: Performance metrics from oscillator simulation
            
        Returns:
            Reward value
        """
        if metrics.get('simulation_failed', False):
            return -10.0  # Heavy penalty for simulation failure
        
        if not metrics['is_oscillating']:
            return -5.0  # Penalty for non-oscillating behavior
        
        # Individual reward components
        period_reward = self._period_reward(metrics['period'])
        amplitude_reward = self._amplitude_reward(metrics['amplitude'])
        stability_reward = metrics['stability']
        oscillation_bonus = 1.0 if metrics['is_oscillating'] else 0.0
        
        # Weighted combination
        total_reward = (
            self.reward_weights['period'] * period_reward +
            self.reward_weights['amplitude'] * amplitude_reward +
            self.reward_weights['stability'] * stability_reward +
            self.reward_weights['oscillation'] * oscillation_bonus
        )
        
        # Bonus for achieving multiple objectives
        if (abs(metrics['period'] - self.target_period) / self.target_period < 0.1 and
            abs(metrics['amplitude'] - self.target_amplitude) / self.target_amplitude < 0.1):
            total_reward += 2.0  # Multi-objective bonus
        
        return total_reward
    
    def _period_reward(self, period: float) -> float:
        """Calculate reward based on period accuracy."""
        if period <= 0:
            return -1.0
        
        error = abs(period - self.target_period) / self.target_period
        return max(-1.0, 1.0 - 2.0 * error)
    
    def _amplitude_reward(self, amplitude: float) -> float:
        """Calculate reward based on amplitude accuracy."""
        if amplitude <= 0:
            return -1.0
        
        error = abs(amplitude - self.target_amplitude) / self.target_amplitude
        return max(-1.0, 1.0 - 2.0 * error)
    
    def _is_goal_reached(self, metrics: Dict) -> bool:
        """Check if optimization goal has been reached."""
        if not metrics['is_oscillating']:
            return False
        
        period_error = abs(metrics['period'] - self.target_period) / self.target_period
        amplitude_error = abs(metrics['amplitude'] - self.target_amplitude) / self.target_amplitude
        
        return (period_error < 0.05 and amplitude_error < 0.05 and 
                metrics['stability'] > 0.8)
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation state.
        
        Returns:
            Observation vector
        """
        # Normalize parameters to [0, 1] range
        normalized_params = []
        for param in self.param_names:
            low, high = self.param_bounds[param]
            normalized_val = (self.current_parameters[param] - low) / (high - low)
            normalized_params.append(normalized_val)
        
        # Get latest metrics (or zeros if first step)
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]
            period = latest_metrics.get('period', 0)
            amplitude = latest_metrics.get('amplitude', 0)
            stability = latest_metrics.get('stability', 0)
            oscillating = 1.0 if latest_metrics.get('is_oscillating', False) else 0.0
        else:
            period = amplitude = stability = oscillating = 0.0
        
        # Combine into observation vector
        observation = np.array(
            normalized_params + 
            [period, amplitude, stability, oscillating, self.target_period, self.target_amplitude],
            dtype=np.float32
        )
        
        return observation
    
    def render(self, mode: str = 'human'):
        """Render the environment (optional visualization)."""
        if mode == 'human':
            if self.metrics_history:
                latest_metrics = self.metrics_history[-1]
                print(f"Step {self.step_count}: Period={latest_metrics.get('period', 0):.3f}, "
                      f"Amplitude={latest_metrics.get('amplitude', 0):.3f}, "
                      f"Stability={latest_metrics.get('stability', 0):.3f}, "
                      f"Reward={self.episode_rewards[-1]:.3f}")
    
    def close(self):
        """Clean up environment resources."""
        pass
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for the current oscillator type."""
        return self.param_bounds.copy()
    
    def get_best_parameters(self) -> Optional[Dict[str, float]]:
        """Get parameters that achieved the best reward."""
        if not self.episode_rewards:
            return None
        
        best_idx = np.argmax(self.episode_rewards)
        return self.parameter_history[best_idx].copy()
    
    def get_training_history(self) -> Dict:
        """Get complete training history for analysis."""
        return {
            'parameters': self.parameter_history.copy(),
            'metrics': self.metrics_history.copy(),
            'rewards': self.episode_rewards.copy(),
            'best_reward': self.best_reward
        }


class NoisyBioEnvironment(BioOscillatorEnvironment):
    """
    Extended environment with dynamic noise and perturbations
    for testing RL robustness to biological variations.
    """
    
    def __init__(self, *args, **kwargs):
        # Extract noise-specific parameters
        self.noise_schedule = kwargs.pop('noise_schedule', None)
        self.perturbation_probability = kwargs.pop('perturbation_probability', 0.1)
        self.perturbation_strength = kwargs.pop('perturbation_strength', 0.2)
        
        super().__init__(*args, **kwargs)
        
        # Dynamic noise tracking
        self.current_noise_level = self.noise_strength
        self.noise_history = []
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute step with dynamic noise and perturbations."""
        
        # Update noise level if schedule is provided
        if self.noise_schedule:
            self.current_noise_level = self._get_scheduled_noise()
        
        # Apply random perturbations
        if np.random.random() < self.perturbation_probability:
            self._apply_perturbation()
        
        # Update noise strength for simulation
        old_noise = self.noise_strength
        self.noise_strength = self.current_noise_level
        
        # Execute normal step
        obs, reward, done, info = super().step(action)
        
        # Restore original noise level
        self.noise_strength = old_noise
        
        # Add noise information to info
        info['noise_level'] = self.current_noise_level
        info['perturbation_applied'] = (np.random.random() < self.perturbation_probability)
        
        self.noise_history.append(self.current_noise_level)
        
        return obs, reward, done, info
    
    def _get_scheduled_noise(self) -> float:
        """Get noise level according to schedule."""
        if callable(self.noise_schedule):
            return self.noise_schedule(self.step_count)
        elif isinstance(self.noise_schedule, (list, np.ndarray)):
            idx = min(self.step_count, len(self.noise_schedule) - 1)
            return self.noise_schedule[idx]
        else:
            return self.noise_strength
    
    def _apply_perturbation(self):
        """Apply random perturbation to current parameters."""
        for param in self.param_names:
            low, high = self.param_bounds[param]
            current_val = self.current_parameters[param]
            
            # Random perturbation
            perturbation_range = self.perturbation_strength * (high - low)
            perturbation = np.random.uniform(-perturbation_range, perturbation_range)
            
            # Apply with bounds checking
            new_val = current_val + perturbation
            self.current_parameters[param] = np.clip(new_val, low, high)
    
    def reset(self) -> np.ndarray:
        """Reset environment with noise tracking."""
        self.noise_history = []
        self.current_noise_level = self.noise_strength
        return super().reset()


# Convenience functions for creating environments
def make_edelstein_env(target_period: float = 10.0, 
                      target_amplitude: float = 2.0,
                      **kwargs) -> BioOscillatorEnvironment:
    """Create Edelstein oscillator environment."""
    return BioOscillatorEnvironment(
        oscillator_type="edelstein",
        target_period=target_period,
        target_amplitude=target_amplitude,
        **kwargs
    )


def make_otero_env(target_period: float = 10.0,
                  target_amplitude: float = 5.0,
                  **kwargs) -> BioOscillatorEnvironment:
    """Create Otero repressilator environment."""
    return BioOscillatorEnvironment(
        oscillator_type="otero",
        target_period=target_period,
        target_amplitude=target_amplitude,
        **kwargs
    )


def make_noisy_env(oscillator_type: str = "edelstein",
                  noise_schedule: Optional[Union[callable, List[float]]] = None,
                  **kwargs) -> NoisyBioEnvironment:
    """Create noisy biological environment for robustness testing."""
    return NoisyBioEnvironment(
        oscillator_type=oscillator_type,
        noise_schedule=noise_schedule,
        **kwargs
    )
