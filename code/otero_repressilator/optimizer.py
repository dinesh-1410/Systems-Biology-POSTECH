"""
Otero Repressilator Optimizer using Reinforcement Learning

This module implements reinforcement learning-based optimization
for the Otero repressilator parameters.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
from .repressilator import OteroRepressilator


class OteroOptimizer:
    """
    Reinforcement Learning Optimizer for Otero Repressilator
    
    Uses a hybrid genetic algorithm with RL-inspired fitness evaluation
    to optimize repressilator parameters for desired oscillatory behavior.
    """
    
    def __init__(self,
                 target_period: float = 10.0,
                 target_amplitude: float = 5.0,
                 target_phase_coherence: float = 0.9,
                 population_size: int = 50,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.8,
                 elite_fraction: float = 0.2):
        """
        Initialize the optimizer.
        
        Args:
            target_period: Desired oscillation period
            target_amplitude: Desired oscillation amplitude
            target_phase_coherence: Desired phase coherence (0-1)
            population_size: Size of the parameter population
            mutation_rate: Probability of parameter mutation
            crossover_rate: Probability of parameter crossover
            elite_fraction: Fraction of best individuals to keep
        """
        self.target_period = target_period
        self.target_amplitude = target_amplitude
        self.target_phase_coherence = target_phase_coherence
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = int(elite_fraction * population_size)
        
        # Parameter bounds for repressilator
        self.param_bounds = {
            'alpha': (50.0, 500.0),      # Maximum transcription rate
            'alpha0': (0.0, 10.0),       # Basal transcription rate
            'beta': (0.1, 1.0),          # Translation rate
            'gamma': (0.5, 5.0),         # Degradation rate
            'n': (1.0, 4.0)              # Hill coefficient
        }
        
        self.population = []
        self.fitness_history = []
        self.best_parameters = None
        self.best_fitness = -np.inf
        
        # RL-inspired adaptive parameters
        self.exploration_rate = 0.3
        self.exploration_decay = 0.99
        self.reward_history = []
        
    def initialize_population(self) -> None:
        """Initialize random population of parameters."""
        self.population = []
        
        for _ in range(self.population_size):
            individual = {}
            for param, (low, high) in self.param_bounds.items():
                individual[param] = np.random.uniform(low, high)
            self.population.append(individual)
    
    def evaluate_fitness(self, parameters: Dict[str, float]) -> float:
        """
        Evaluate fitness using multi-objective RL-inspired reward function.
        
        Args:
            parameters: Parameter dictionary
            
        Returns:
            Fitness score (higher is better)
        """
        try:
            # Create and simulate repressilator
            repressilator = OteroRepressilator(**parameters)
            repressilator.simulate(time_span=(0, 100), num_points=2000)
            
            metrics = repressilator.get_fitness_metrics()
            
            if not metrics['is_oscillating']:
                return -100.0  # Heavy penalty for non-oscillating behavior
            
            period = metrics['period']
            amplitude = metrics['amplitude']
            phase_coherence = metrics['phase_coherence']
            stability = metrics['stability']
            
            # Multi-objective reward components
            period_reward = self._calculate_period_reward(period)
            amplitude_reward = self._calculate_amplitude_reward(amplitude)
            phase_reward = self._calculate_phase_reward(phase_coherence)
            stability_reward = stability
            
            # Adaptive weighting based on optimization progress
            weights = self._get_adaptive_weights()
            
            # Combined fitness with adaptive weights
            fitness = (
                weights['period'] * period_reward + 
                weights['amplitude'] * amplitude_reward + 
                weights['phase'] * phase_reward +
                weights['stability'] * stability_reward
            )
            
            # Bonus for achieving multiple objectives simultaneously
            if (abs(period - self.target_period) / self.target_period < 0.1 and
                abs(amplitude - self.target_amplitude) / self.target_amplitude < 0.1 and
                phase_coherence > 0.8):
                fitness += 0.5  # Bonus for multi-objective success
            
            return fitness
            
        except Exception as e:
            # Penalty for simulation failures
            return -100.0
    
    def _calculate_period_reward(self, period: float) -> float:
        """Calculate period-specific reward with tolerance."""
        if period <= 0:
            return -1.0
        
        error = abs(period - self.target_period) / self.target_period
        if error < 0.05:  # Very close to target
            return 1.0
        elif error < 0.2:  # Reasonably close
            return 1.0 - 2 * error
        else:  # Far from target
            return -error
    
    def _calculate_amplitude_reward(self, amplitude: float) -> float:
        """Calculate amplitude-specific reward with tolerance."""
        if amplitude <= 0:
            return -1.0
        
        error = abs(amplitude - self.target_amplitude) / self.target_amplitude
        if error < 0.1:  # Very close to target
            return 1.0
        elif error < 0.3:  # Reasonably close
            return 1.0 - error
        else:  # Far from target
            return -0.5 * error
    
    def _calculate_phase_reward(self, phase_coherence: float) -> float:
        """Calculate phase coherence reward."""
        if phase_coherence >= self.target_phase_coherence:
            return 1.0
        else:
            return phase_coherence / self.target_phase_coherence - 0.5
    
    def _get_adaptive_weights(self) -> Dict[str, float]:
        """Get adaptive weights based on optimization progress."""
        if len(self.fitness_history) < 10:
            # Early stage: focus on basic oscillation
            return {'period': 0.4, 'amplitude': 0.3, 'phase': 0.2, 'stability': 0.1}
        
        # Later stages: balance all objectives
        recent_fitness = [h['best_fitness'] for h in self.fitness_history[-10:]]
        improvement = recent_fitness[-1] - recent_fitness[0] if len(recent_fitness) > 1 else 0
        
        if improvement < 0.01:  # Stagnation: increase exploration
            self.exploration_rate = min(0.5, self.exploration_rate * 1.1)
            return {'period': 0.3, 'amplitude': 0.3, 'phase': 0.3, 'stability': 0.1}
        else:  # Progress: maintain balance
            return {'period': 0.35, 'amplitude': 0.35, 'phase': 0.25, 'stability': 0.05}
    
    def selection(self, fitness_scores: List[float]) -> List[int]:
        """
        Select parents using adaptive tournament selection.
        
        Args:
            fitness_scores: List of fitness scores
            
        Returns:
            Indices of selected parents
        """
        selected_indices = []
        
        # Keep elite individuals
        sorted_indices = np.argsort(fitness_scores)[::-1]
        selected_indices.extend(sorted_indices[:self.elite_size])
        
        # Adaptive tournament selection for remaining individuals
        tournament_size = max(2, int(3 * (1 - self.exploration_rate)))
        
        while len(selected_indices) < self.population_size:
            if np.random.random() < self.exploration_rate:
                # Exploration: random selection
                selected_indices.append(np.random.randint(len(fitness_scores)))
            else:
                # Exploitation: tournament selection
                tournament_indices = np.random.choice(
                    len(fitness_scores), tournament_size, replace=False
                )
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                selected_indices.append(winner_idx)
        
        return selected_indices
    
    def crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Create offspring through adaptive crossover.
        
        Args:
            parent1, parent2: Parent parameter dictionaries
            
        Returns:
            Two offspring parameter dictionaries
        """
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        offspring1, offspring2 = {}, {}
        
        # Blend crossover for continuous parameters
        for param in self.param_bounds.keys():
            alpha = np.random.uniform(-0.1, 1.1)  # Blend factor
            val1 = parent1[param]
            val2 = parent2[param]
            
            new_val1 = alpha * val1 + (1 - alpha) * val2
            new_val2 = alpha * val2 + (1 - alpha) * val1
            
            # Ensure bounds
            low, high = self.param_bounds[param]
            offspring1[param] = np.clip(new_val1, low, high)
            offspring2[param] = np.clip(new_val2, low, high)
        
        return offspring1, offspring2
    
    def mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """
        Mutate individual parameters with adaptive mutation.
        
        Args:
            individual: Parameter dictionary to mutate
            
        Returns:
            Mutated parameter dictionary
        """
        mutated = individual.copy()
        
        # Adaptive mutation rate based on exploration
        effective_mutation_rate = self.mutation_rate * (1 + self.exploration_rate)
        
        for param, (low, high) in self.param_bounds.items():
            if np.random.random() < effective_mutation_rate:
                current_val = mutated[param]
                
                # Adaptive mutation strength
                mutation_strength = 0.1 * (high - low) * (1 + self.exploration_rate)
                
                if np.random.random() < 0.7:
                    # Gaussian mutation
                    new_val = current_val + np.random.normal(0, mutation_strength)
                else:
                    # Uniform mutation (for exploration)
                    new_val = np.random.uniform(low, high)
                
                # Ensure bounds
                mutated[param] = np.clip(new_val, low, high)
        
        return mutated
    
    def optimize_parameters(self, generations: int = 150, verbose: bool = True) -> Dict[str, float]:
        """
        Optimize repressilator parameters using adaptive genetic algorithm.
        
        Args:
            generations: Number of generations to evolve
            verbose: Whether to print progress
            
        Returns:
            Best parameter set found
        """
        if verbose:
            print("Starting Otero Repressilator Optimization...")
            print(f"Targets: Period={self.target_period}, Amplitude={self.target_amplitude}, Phase Coherence={self.target_phase_coherence}")
        
        # Initialize population
        self.initialize_population()
        self.fitness_history = []
        self.reward_history = []
        
        for generation in range(generations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            for individual in self.population:
                fitness = self.evaluate_fitness(individual)
                fitness_scores.append(fitness)
            
            # Track best individual
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_parameters = self.population[best_idx].copy()
            
            # Update exploration rate
            self.exploration_rate *= self.exploration_decay
            
            # Track progress
            generation_stats = {
                'generation': generation,
                'best_fitness': max(fitness_scores),
                'mean_fitness': np.mean(fitness_scores),
                'std_fitness': np.std(fitness_scores),
                'exploration_rate': self.exploration_rate
            }
            self.fitness_history.append(generation_stats)
            self.reward_history.append(max(fitness_scores))
            
            if verbose and generation % 20 == 0:
                print(f"Generation {generation}: Best fitness = {max(fitness_scores):.4f}, "
                      f"Exploration = {self.exploration_rate:.3f}")
            
            # Selection
            selected_indices = self.selection(fitness_scores)
            
            # Create new population
            new_population = []
            
            # Add elite individuals
            for i in range(self.elite_size):
                new_population.append(self.population[selected_indices[i]].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1_idx = np.random.choice(selected_indices)
                parent2_idx = np.random.choice(selected_indices)
                
                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]
                
                offspring1, offspring2 = self.crossover(parent1, parent2)
                
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                
                new_population.extend([offspring1, offspring2])
            
            # Trim to exact population size
            self.population = new_population[:self.population_size]
        
        if verbose:
            print(f"Optimization completed! Best fitness: {self.best_fitness:.4f}")
            print("Best parameters:")
            for param, value in self.best_parameters.items():
                print(f"  {param}: {value:.4f}")
        
        return self.best_parameters
    
    def plot_optimization_progress(self, save_path: Optional[str] = None) -> None:
        """
        Plot optimization progress with multiple metrics.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.fitness_history:
            print("No optimization history available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        generations = [h['generation'] for h in self.fitness_history]
        best_fitness = [h['best_fitness'] for h in self.fitness_history]
        mean_fitness = [h['mean_fitness'] for h in self.fitness_history]
        exploration_rates = [h['exploration_rate'] for h in self.fitness_history]
        
        # Fitness evolution
        axes[0, 0].plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness')
        axes[0, 0].plot(generations, mean_fitness, 'r--', linewidth=2, label='Mean Fitness')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Fitness Score')
        axes[0, 0].set_title('Fitness Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Exploration rate
        axes[0, 1].plot(generations, exploration_rates, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Exploration Rate')
        axes[0, 1].set_title('Adaptive Exploration')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reward distribution (recent generations)
        if len(self.reward_history) > 50:
            recent_rewards = self.reward_history[-50:]
            axes[1, 0].hist(recent_rewards, bins=20, alpha=0.7, color='orange')
            axes[1, 0].set_xlabel('Fitness Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Recent Fitness Distribution')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Convergence analysis
        if len(best_fitness) > 20:
            window_size = 20
            smoothed_fitness = np.convolve(best_fitness, np.ones(window_size)/window_size, mode='valid')
            smoothed_generations = generations[window_size-1:]
            axes[1, 1].plot(smoothed_generations, smoothed_fitness, 'purple', linewidth=2)
            axes[1, 1].set_xlabel('Generation')
            axes[1, 1].set_ylabel('Smoothed Best Fitness')
            axes[1, 1].set_title('Convergence Trend')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def validate_best_solution(self) -> Dict:
        """
        Validate the best solution with extended analysis.
        
        Returns:
            Dictionary with comprehensive validation results
        """
        if self.best_parameters is None:
            return {"error": "No optimization performed yet"}
        
        repressilator = OteroRepressilator(**self.best_parameters)
        result = repressilator.simulate(time_span=(0, 150), num_points=3000)
        metrics = repressilator.get_fitness_metrics()
        
        validation = {
            'parameters': self.best_parameters.copy(),
            'fitness': self.best_fitness,
            'achieved_period': metrics['period'],
            'achieved_amplitude': metrics['amplitude'],
            'phase_coherence': metrics['phase_coherence'],
            'stability': metrics['stability'],
            'phase_differences': metrics['phase_differences'],
            'is_oscillating': metrics['is_oscillating'],
            'period_error': abs(metrics['period'] - self.target_period) / self.target_period if metrics['period'] else 1.0,
            'amplitude_error': abs(metrics['amplitude'] - self.target_amplitude) / self.target_amplitude if metrics['amplitude'] else 1.0,
            'phase_error': abs(metrics['phase_coherence'] - self.target_phase_coherence),
            'simulation_data': result
        }
        
        return validation
    
    def plot_best_solution(self, save_path: Optional[str] = None) -> None:
        """
        Plot comprehensive analysis of the best solution.
        
        Args:
            save_path: Optional path to save the plot
        """
        validation = self.validate_best_solution()
        
        if "error" in validation:
            print(validation["error"])
            return
        
        # Create repressilator with best parameters for plotting
        repressilator = OteroRepressilator(**self.best_parameters)
        repressilator.simulate(time_span=(0, 100), num_points=2000)
        
        # Use the repressilator's plotting method
        repressilator.plot_trajectory(save_path=save_path)
        
        # Print validation summary
        print("\n=== Optimization Results Summary ===")
        print(f"Target Period: {self.target_period:.2f}, Achieved: {validation['achieved_period']:.2f} (Error: {validation['period_error']:.1%})")
        print(f"Target Amplitude: {self.target_amplitude:.2f}, Achieved: {validation['achieved_amplitude']:.2f} (Error: {validation['amplitude_error']:.1%})")
        print(f"Target Phase Coherence: {self.target_phase_coherence:.2f}, Achieved: {validation['phase_coherence']:.2f}")
        print(f"Stability: {validation['stability']:.3f}")
        print(f"Phase Differences: {validation['phase_differences']}")
    
    def analyze_parameter_sensitivity(self, param_name: str, 
                                    deviation_percent: float = 20.0,
                                    num_samples: int = 20) -> Dict:
        """
        Analyze sensitivity of the optimized solution to parameter variations.
        
        Args:
            param_name: Parameter to analyze
            deviation_percent: Percentage deviation from optimized value
            num_samples: Number of samples around the optimized value
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        if self.best_parameters is None:
            return {"error": "No optimization performed yet"}
        
        if param_name not in self.best_parameters:
            return {"error": f"Unknown parameter: {param_name}"}
        
        optimal_value = self.best_parameters[param_name]
        deviation = optimal_value * deviation_percent / 100.0
        
        param_values = np.linspace(optimal_value - deviation, 
                                 optimal_value + deviation, 
                                 num_samples)
        
        results = {
            'parameter': param_name,
            'optimal_value': optimal_value,
            'param_values': param_values,
            'fitness_scores': [],
            'periods': [],
            'amplitudes': [],
            'phase_coherences': []
        }
        
        for param_val in param_values:
            # Create modified parameters
            test_params = self.best_parameters.copy()
            test_params[param_name] = param_val
            
            # Evaluate fitness
            fitness = self.evaluate_fitness(test_params)
            results['fitness_scores'].append(fitness)
            
            # Get detailed metrics
            try:
                repressilator = OteroRepressilator(**test_params)
                repressilator.simulate(time_span=(0, 100), num_points=2000)
                metrics = repressilator.get_fitness_metrics()
                
                results['periods'].append(metrics['period'] if metrics['period'] else 0)
                results['amplitudes'].append(metrics['amplitude'] if metrics['amplitude'] else 0)
                results['phase_coherences'].append(metrics['phase_coherence'])
                
            except Exception:
                results['periods'].append(0)
                results['amplitudes'].append(0)
                results['phase_coherences'].append(0)
        
        return results
