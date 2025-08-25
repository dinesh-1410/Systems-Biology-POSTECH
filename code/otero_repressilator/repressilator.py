"""
Otero Repressilator Implementation

This module implements the Otero repressilator model,
a genetic circuit oscillator based on three mutually repressing genes.
"""

import numpy as np
from scipy.integrate import odeint
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class OteroRepressilator:
    """
    Otero Repressilator Model
    
    Implements the three-gene repressilator circuit where each gene
    represses the next in a cyclic manner: Gene1 -> Gene2 -> Gene3 -> Gene1
    """
    
    def __init__(self,
                 alpha: float = 216.4,
                 alpha0: float = 0.0,
                 beta: float = 0.2,
                 gamma: float = 2.0,
                 n: float = 2.0,
                 noise_strength: float = 0.0):
        """
        Initialize the Otero repressilator with parameters.
        
        Args:
            alpha: Maximum transcription rate
            alpha0: Basal transcription rate
            beta: Translation rate
            gamma: Degradation rate
            n: Hill coefficient (cooperativity)
            noise_strength: Strength of biological noise
        """
        self.parameters = {
            'alpha': alpha,
            'alpha0': alpha0,
            'beta': beta,
            'gamma': gamma,
            'n': n
        }
        self.noise_strength = noise_strength
        self.state_history = []
        self.time_points = []
        
    def hill_function(self, x: float, n: float) -> float:
        """
        Hill function for gene repression.
        
        Args:
            x: Repressor concentration
            n: Hill coefficient
            
        Returns:
            Repression factor
        """
        return 1.0 / (1.0 + x**n)
    
    def ode_system(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Define the ODE system for the Otero repressilator.
        
        Args:
            state: Current state [m1, m2, m3, p1, p2, p3]
                  where m = mRNA, p = protein
            t: Current time
            
        Returns:
            derivatives: [dm1/dt, dm2/dt, dm3/dt, dp1/dt, dp2/dt, dp3/dt]
        """
        m1, m2, m3, p1, p2, p3 = state
        p = self.parameters
        
        # mRNA dynamics (transcription - degradation)
        dm1_dt = (p['alpha'] * self.hill_function(p3, p['n']) + p['alpha0']) - p['gamma'] * m1
        dm2_dt = (p['alpha'] * self.hill_function(p1, p['n']) + p['alpha0']) - p['gamma'] * m2
        dm3_dt = (p['alpha'] * self.hill_function(p2, p['n']) + p['alpha0']) - p['gamma'] * m3
        
        # Protein dynamics (translation - degradation)
        dp1_dt = p['beta'] * m1 - p['gamma'] * p1
        dp2_dt = p['beta'] * m2 - p['gamma'] * p2
        dp3_dt = p['beta'] * m3 - p['gamma'] * p3
        
        derivatives = np.array([dm1_dt, dm2_dt, dm3_dt, dp1_dt, dp2_dt, dp3_dt])
        
        # Add biological noise if specified
        if self.noise_strength > 0:
            noise = np.random.normal(0, self.noise_strength, 6)
            derivatives += noise
            
        return derivatives
    
    def simulate(self, 
                 initial_state: Optional[Tuple] = None,
                 time_span: Tuple[float, float] = (0, 50),
                 num_points: int = 1000) -> Dict:
        """
        Simulate the repressilator dynamics.
        
        Args:
            initial_state: Initial conditions [m1, m2, m3, p1, p2, p3]
            time_span: Time span for simulation
            num_points: Number of time points
            
        Returns:
            Dictionary with simulation results
        """
        if initial_state is None:
            # Default initial conditions with small asymmetry
            initial_state = (1.0, 0.5, 0.1, 2.0, 1.0, 0.5)
            
        self.time_points = np.linspace(time_span[0], time_span[1], num_points)
        
        # Solve the ODE system
        solution = odeint(self.ode_system, initial_state, self.time_points)
        
        self.state_history = solution
        
        return {
            'time': self.time_points,
            'm1': solution[:, 0], 'm2': solution[:, 1], 'm3': solution[:, 2],
            'p1': solution[:, 3], 'p2': solution[:, 4], 'p3': solution[:, 5],
            'parameters': self.parameters.copy()
        }
    
    def calculate_period(self, species: str = 'p1', threshold_fraction: float = 0.1) -> Optional[float]:
        """
        Calculate the oscillation period using peak detection.
        
        Args:
            species: Which species to analyze ('m1', 'm2', 'm3', 'p1', 'p2', 'p3')
            threshold_fraction: Fraction of max amplitude for peak detection
            
        Returns:
            Period of oscillation or None if no oscillation detected
        """
        if len(self.state_history) == 0:
            return None
        
        species_map = {'m1': 0, 'm2': 1, 'm3': 2, 'p1': 3, 'p2': 4, 'p3': 5}
        if species not in species_map:
            raise ValueError(f"Unknown species: {species}")
            
        X = self.state_history[:, species_map[species]]
        
        # Find peaks
        max_val = np.max(X)
        min_val = np.min(X)
        threshold = min_val + threshold_fraction * (max_val - min_val)
        
        peaks = []
        for i in range(1, len(X) - 1):
            if X[i] > X[i-1] and X[i] > X[i+1] and X[i] > threshold:
                peaks.append(i)
        
        if len(peaks) < 2:
            return None
            
        # Calculate average period
        periods = []
        for i in range(1, len(peaks)):
            period = self.time_points[peaks[i]] - self.time_points[peaks[i-1]]
            periods.append(period)
            
        return np.mean(periods) if periods else None
    
    def calculate_amplitude(self, species: str = 'p1') -> Optional[float]:
        """
        Calculate the oscillation amplitude.
        
        Args:
            species: Which species to analyze
            
        Returns:
            Amplitude of oscillation or None if no simulation data
        """
        if len(self.state_history) == 0:
            return None
        
        species_map = {'m1': 0, 'm2': 1, 'm3': 2, 'p1': 3, 'p2': 4, 'p3': 5}
        if species not in species_map:
            raise ValueError(f"Unknown species: {species}")
            
        X = self.state_history[:, species_map[species]]
        return (np.max(X) - np.min(X)) / 2
    
    def calculate_phase_differences(self) -> Dict[str, float]:
        """
        Calculate phase differences between the three genes.
        
        Returns:
            Dictionary with phase differences
        """
        if len(self.state_history) == 0:
            return {}
        
        # Calculate phase for each protein using Hilbert transform
        from scipy.signal import hilbert
        
        phases = {}
        for i, protein in enumerate(['p1', 'p2', 'p3']):
            signal = self.state_history[:, 3 + i]  # protein concentrations
            analytic_signal = hilbert(signal - np.mean(signal))
            phases[protein] = np.angle(analytic_signal)
        
        # Calculate phase differences (should be ~120 degrees for ideal repressilator)
        phase_diffs = {
            'p1_p2': np.mean(phases['p2'] - phases['p1']),
            'p2_p3': np.mean(phases['p3'] - phases['p2']),
            'p3_p1': np.mean(phases['p1'] - phases['p3'])
        }
        
        # Convert to degrees and normalize to [0, 360)
        for key in phase_diffs:
            phase_diffs[key] = (np.degrees(phase_diffs[key]) % 360)
        
        return phase_diffs
    
    def is_oscillating(self, min_periods: int = 2) -> bool:
        """
        Check if the system is oscillating.
        
        Args:
            min_periods: Minimum number of periods to consider oscillating
            
        Returns:
            True if oscillating, False otherwise
        """
        period = self.calculate_period()
        if period is None:
            return False
            
        total_time = self.time_points[-1] - self.time_points[0]
        num_periods = total_time / period
        
        return num_periods >= min_periods
    
    def plot_trajectory(self, save_path: Optional[str] = None) -> None:
        """
        Plot the repressilator trajectory.
        
        Args:
            save_path: Optional path to save the plot
        """
        if len(self.state_history) == 0:
            print("No simulation data available. Run simulate() first.")
            return
            
        fig = plt.figure(figsize=(15, 10))
        
        # Time series plot for proteins
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(self.time_points, self.state_history[:, 3], 'b-', linewidth=2, label='Protein 1')
        ax1.plot(self.time_points, self.state_history[:, 4], 'r-', linewidth=2, label='Protein 2')
        ax1.plot(self.time_points, self.state_history[:, 5], 'g-', linewidth=2, label='Protein 3')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Protein Concentration')
        ax1.set_title('Protein Time Series')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Time series plot for mRNAs
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(self.time_points, self.state_history[:, 0], 'b--', linewidth=2, label='mRNA 1')
        ax2.plot(self.time_points, self.state_history[:, 1], 'r--', linewidth=2, label='mRNA 2')
        ax2.plot(self.time_points, self.state_history[:, 2], 'g--', linewidth=2, label='mRNA 3')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('mRNA Concentration')
        ax2.set_title('mRNA Time Series')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3D phase portrait
        ax3 = plt.subplot(2, 3, 3, projection='3d')
        ax3.plot(self.state_history[:, 3], self.state_history[:, 4], self.state_history[:, 5], 'purple', linewidth=2)
        ax3.set_xlabel('Protein 1')
        ax3.set_ylabel('Protein 2')
        ax3.set_zlabel('Protein 3')
        ax3.set_title('3D Phase Portrait')
        
        # 2D projections
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(self.state_history[:, 3], self.state_history[:, 4], 'purple', linewidth=2)
        ax4.set_xlabel('Protein 1')
        ax4.set_ylabel('Protein 2')
        ax4.set_title('P1 vs P2 Phase Plot')
        ax4.grid(True, alpha=0.3)
        
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(self.state_history[:, 4], self.state_history[:, 5], 'orange', linewidth=2)
        ax5.set_xlabel('Protein 2')
        ax5.set_ylabel('Protein 3')
        ax5.set_title('P2 vs P3 Phase Plot')
        ax5.grid(True, alpha=0.3)
        
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(self.state_history[:, 5], self.state_history[:, 3], 'brown', linewidth=2)
        ax6.set_xlabel('Protein 3')
        ax6.set_ylabel('Protein 1')
        ax6.set_title('P3 vs P1 Phase Plot')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def update_parameters(self, new_parameters: Dict[str, float]) -> None:
        """
        Update repressilator parameters.
        
        Args:
            new_parameters: Dictionary of parameter updates
        """
        for key, value in new_parameters.items():
            if key in self.parameters:
                self.parameters[key] = value
            else:
                raise ValueError(f"Unknown parameter: {key}")
    
    def get_fitness_metrics(self) -> Dict:
        """
        Calculate fitness metrics for optimization.
        
        Returns:
            Dictionary of fitness metrics
        """
        if len(self.state_history) == 0:
            return {'period': 0, 'amplitude': 0, 'stability': 0, 'phase_coherence': 0}
        
        period = self.calculate_period()
        amplitude = self.calculate_amplitude()
        phase_diffs = self.calculate_phase_differences()
        
        # Stability measure (inverse of coefficient of variation)
        p1 = self.state_history[:, 3]
        if len(p1) > 100:  # Use last part of simulation for stability
            p1_stable = p1[-100:]
            cv = np.std(p1_stable) / (np.mean(p1_stable) + 1e-6)
            stability = 1.0 / (1.0 + cv)
        else:
            stability = 0.0
        
        # Phase coherence (how close to 120° phase differences)
        if phase_diffs:
            ideal_phase_diff = 120.0  # degrees
            phase_errors = []
            for diff in phase_diffs.values():
                error = min(abs(diff - ideal_phase_diff), 
                          abs(diff - (ideal_phase_diff + 360)),
                          abs(diff - (ideal_phase_diff - 360)))
                phase_errors.append(error)
            phase_coherence = 1.0 / (1.0 + np.mean(phase_errors) / 120.0)
        else:
            phase_coherence = 0.0
        
        return {
            'period': period if period else 0,
            'amplitude': amplitude if amplitude else 0,
            'stability': stability,
            'phase_coherence': phase_coherence,
            'phase_differences': phase_diffs,
            'is_oscillating': self.is_oscillating()
        }
    
    def analyze_bifurcation(self, param_name: str, param_range: Tuple[float, float], 
                          num_points: int = 50) -> Dict:
        """
        Analyze bifurcation behavior as a function of a parameter.
        
        Args:
            param_name: Name of parameter to vary
            param_range: (min, max) range for parameter
            num_points: Number of parameter values to test
            
        Returns:
            Dictionary with bifurcation analysis results
        """
        if param_name not in self.parameters:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        original_value = self.parameters[param_name]
        param_values = np.linspace(param_range[0], param_range[1], num_points)
        
        results = {
            'parameter': param_name,
            'param_values': param_values,
            'periods': [],
            'amplitudes': [],
            'is_oscillating': [],
            'max_concentrations': [],
            'min_concentrations': []
        }
        
        for param_val in param_values:
            self.parameters[param_name] = param_val
            
            try:
                # Simulate with current parameter value
                self.simulate(time_span=(0, 100), num_points=2000)
                metrics = self.get_fitness_metrics()
                
                results['periods'].append(metrics['period'] if metrics['period'] else 0)
                results['amplitudes'].append(metrics['amplitude'] if metrics['amplitude'] else 0)
                results['is_oscillating'].append(metrics['is_oscillating'])
                
                # Track max and min concentrations for bifurcation diagram
                p1 = self.state_history[-500:, 3]  # Last part of simulation
                results['max_concentrations'].append(np.max(p1))
                results['min_concentrations'].append(np.min(p1))
                
            except Exception:
                # Handle simulation failures
                results['periods'].append(0)
                results['amplitudes'].append(0)
                results['is_oscillating'].append(False)
                results['max_concentrations'].append(0)
                results['min_concentrations'].append(0)
        
        # Restore original parameter value
        self.parameters[param_name] = original_value
        
        return results
    
    def plot_bifurcation_diagram(self, bifurcation_data: Dict, save_path: Optional[str] = None) -> None:
        """
        Plot bifurcation diagram.
        
        Args:
            bifurcation_data: Output from analyze_bifurcation
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        param_values = bifurcation_data['param_values']
        max_conc = bifurcation_data['max_concentrations']
        min_conc = bifurcation_data['min_concentrations']
        
        # Bifurcation diagram
        ax1.plot(param_values, max_conc, 'r.', markersize=2, alpha=0.7, label='Max')
        ax1.plot(param_values, min_conc, 'b.', markersize=2, alpha=0.7, label='Min')
        ax1.set_xlabel(f'{bifurcation_data["parameter"]}')
        ax1.set_ylabel('Protein 1 Concentration')
        ax1.set_title('Bifurcation Diagram')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Period vs parameter
        periods = bifurcation_data['periods']
        oscillating_mask = [p > 0 for p in periods]
        osc_params = [p for i, p in enumerate(param_values) if oscillating_mask[i]]
        osc_periods = [p for i, p in enumerate(periods) if oscillating_mask[i]]
        
        ax2.plot(osc_params, osc_periods, 'go', markersize=4, label='Oscillating')
        ax2.set_xlabel(f'{bifurcation_data["parameter"]}')
        ax2.set_ylabel('Period')
        ax2.set_title('Period vs Parameter')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
