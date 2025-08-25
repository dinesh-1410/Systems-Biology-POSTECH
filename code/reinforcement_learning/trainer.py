"""
Reinforcement Learning Trainer for Biological Oscillator Optimization

This module implements training frameworks for RL agents to optimize
synthetic biological oscillators with advanced features like curriculum
learning, noise adaptation, and bifurcation handling.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Callable
import json
from pathlib import Path
import time
from tqdm import tqdm

from .agent import DDPG_BioAgent, PPO_BioAgent
from .environment import BioOscillatorEnvironment, NoisyBioEnvironment


class RLTrainer:
    """
    Advanced trainer for RL agents optimizing biological oscillators.
    
    Features:
    - Curriculum learning with progressive difficulty
    - Adaptive noise scheduling
    - Multi-objective optimization
    - Bifurcation robustness training
    - Comprehensive logging and visualization
    """
    
    def __init__(self,
                 agent,
                 environment,
                 max_episodes: int = 1000,
                 max_steps_per_episode: int = 100,
                 evaluation_frequency: int = 50,
                 save_frequency: int = 100,
                 log_dir: str = "logs",
                 model_dir: str = "models"):
        """
        Initialize the RL trainer.
        
        Args:
            agent: RL agent to train
            environment: Training environment
            max_episodes: Maximum number of training episodes
            max_steps_per_episode: Maximum steps per episode
            evaluation_frequency: How often to evaluate agent
            save_frequency: How often to save model
            log_dir: Directory for logs
            model_dir: Directory for saved models
        """
        self.agent = agent
        self.env = environment
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.evaluation_frequency = evaluation_frequency
        self.save_frequency = save_frequency
        
        # Create directories
        self.log_dir = Path(log_dir)
        self.model_dir = Path(model_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': [],
            'parameter_evolution': [],
            'noise_levels': [],
            'evaluation_scores': [],
            'best_parameters': None,
            'best_reward': -np.inf
        }
        
        # Curriculum learning
        self.curriculum_enabled = False
        self.curriculum_stages = []
        self.current_stage = 0
        
        # Evaluation environment
        self.eval_env = None
        
    def enable_curriculum_learning(self, stages: List[Dict]):
        """
        Enable curriculum learning with progressive stages.
        
        Args:
            stages: List of curriculum stages with environment parameters
        """
        self.curriculum_enabled = True
        self.curriculum_stages = stages
        self.current_stage = 0
        print(f"Curriculum learning enabled with {len(stages)} stages")
    
    def set_evaluation_environment(self, eval_env):
        """Set separate environment for evaluation."""
        self.eval_env = eval_env
    
    def train(self, verbose: bool = True) -> Dict:
        """
        Train the RL agent with advanced features.
        
        Args:
            verbose: Whether to print training progress
            
        Returns:
            Training statistics dictionary
        """
        if verbose:
            print("Starting RL training for biological oscillator optimization...")
            print(f"Agent: {type(self.agent).__name__}")
            print(f"Environment: {type(self.env).__name__}")
            print(f"Max episodes: {self.max_episodes}")
        
        start_time = time.time()
        episode_rewards = []
        success_count = 0
        
        # Training loop
        for episode in tqdm(range(self.max_episodes), desc="Training"):
            
            # Update curriculum if enabled
            if self.curriculum_enabled:
                self._update_curriculum(episode)
            
            # Run episode
            episode_reward, episode_length, episode_info = self._run_episode(episode)
            episode_rewards.append(episode_reward)
            
            # Track success
            if episode_info.get('goal_reached', False):
                success_count += 1
            
            # Update training statistics
            self._update_training_stats(episode, episode_reward, episode_length, episode_info)
            
            # Evaluation
            if (episode + 1) % self.evaluation_frequency == 0:
                eval_score = self._evaluate_agent()
                self.training_stats['evaluation_scores'].append({
                    'episode': episode + 1,
                    'score': eval_score
                })
                
                if verbose:
                    recent_rewards = episode_rewards[-self.evaluation_frequency:]
                    avg_reward = np.mean(recent_rewards)
                    success_rate = success_count / self.evaluation_frequency
                    print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.3f}, "
                          f"Success Rate = {success_rate:.2%}, "
                          f"Eval Score = {eval_score:.3f}")
                    success_count = 0
            
            # Save model
            if (episode + 1) % self.save_frequency == 0:
                self._save_checkpoint(episode + 1)
        
        training_time = time.time() - start_time
        
        if verbose:
            print(f"Training completed in {training_time:.2f} seconds")
            print(f"Best reward achieved: {self.training_stats['best_reward']:.3f}")
        
        # Save final results
        self._save_training_results()
        
        return self.training_stats
    
    def _run_episode(self, episode: int) -> Tuple[float, int, Dict]:
        """Run a single training episode."""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_info = {}
        
        for step in range(self.max_steps_per_episode):
            # Select action
            if isinstance(self.agent, DDPG_BioAgent):
                action = self.agent.act(state, add_noise=True)
            elif isinstance(self.agent, PPO_BioAgent):
                action, log_prob, value = self.agent.act(state)
            else:
                action = self.agent.act(state)
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience and update agent
            if isinstance(self.agent, DDPG_BioAgent):
                self.agent.update(state, action, reward, next_state, done)
            elif isinstance(self.agent, PPO_BioAgent):
                self.agent.store_experience(state, action, reward, log_prob, value, done)
            
            # Update episode statistics
            episode_reward += reward
            episode_length += 1
            episode_info.update(info)
            
            # Check for early termination
            if done:
                break
            
            state = next_state
        
        # PPO learning at end of episode
        if isinstance(self.agent, PPO_BioAgent):
            self.agent.learn()
        
        return episode_reward, episode_length, episode_info
    
    def _update_curriculum(self, episode: int):
        """Update curriculum learning stage."""
        if self.current_stage < len(self.curriculum_stages) - 1:
            stage_info = self.curriculum_stages[self.current_stage]
            episodes_per_stage = stage_info.get('episodes', 100)
            
            if episode > 0 and episode % episodes_per_stage == 0:
                self.current_stage += 1
                new_stage = self.curriculum_stages[self.current_stage]
                
                # Update environment parameters
                for param, value in new_stage.get('env_params', {}).items():
                    if hasattr(self.env, param):
                        setattr(self.env, param, value)
                
                print(f"Curriculum: Advanced to stage {self.current_stage + 1}")
    
    def _update_training_stats(self, episode: int, reward: float, length: int, info: Dict):
        """Update training statistics."""
        self.training_stats['episode_rewards'].append(reward)
        self.training_stats['episode_lengths'].append(length)
        
        # Track best performance
        if reward > self.training_stats['best_reward']:
            self.training_stats['best_reward'] = reward
            self.training_stats['best_parameters'] = info.get('parameters', {})
        
        # Track parameter evolution
        if 'parameters' in info:
            self.training_stats['parameter_evolution'].append({
                'episode': episode,
                'parameters': info['parameters'].copy()
            })
        
        # Track noise levels for noisy environments
        if 'noise_level' in info:
            self.training_stats['noise_levels'].append(info['noise_level'])
        
        # Calculate rolling success rate
        if len(self.training_stats['episode_rewards']) >= 100:
            recent_episodes = self.training_stats['episode_rewards'][-100:]
            # Define success as achieving reward > threshold
            success_threshold = 0.5  # Adjust based on reward scale
            success_rate = sum(r > success_threshold for r in recent_episodes) / len(recent_episodes)
            self.training_stats['success_rate'].append(success_rate)
    
    def _evaluate_agent(self, num_episodes: int = 10) -> float:
        """Evaluate agent performance on separate episodes."""
        eval_env = self.eval_env if self.eval_env else self.env
        total_reward = 0
        
        for _ in range(num_episodes):
            state = eval_env.reset()
            episode_reward = 0
            
            for _ in range(self.max_steps_per_episode):
                if isinstance(self.agent, DDPG_BioAgent):
                    action = self.agent.act(state, add_noise=False)  # No exploration during eval
                elif isinstance(self.agent, PPO_BioAgent):
                    action, _, _ = self.agent.act(state)
                else:
                    action = self.agent.act(state)
                
                next_state, reward, done, _ = eval_env.step(action)
                episode_reward += reward
                
                if done:
                    break
                
                state = next_state
            
            total_reward += episode_reward
        
        return total_reward / num_episodes
    
    def _save_checkpoint(self, episode: int):
        """Save model checkpoint."""
        checkpoint_path = self.model_dir / f"checkpoint_episode_{episode}.pth"
        self.agent.save_model(str(checkpoint_path))
        
        # Save training statistics
        stats_path = self.log_dir / f"training_stats_episode_{episode}.json"
        with open(stats_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_stats = {}
            for key, value in self.training_stats.items():
                if isinstance(value, np.ndarray):
                    json_stats[key] = value.tolist()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    json_stats[key] = [v.tolist() for v in value]
                else:
                    json_stats[key] = value
            json.dump(json_stats, f, indent=2)
    
    def _save_training_results(self):
        """Save final training results."""
        results_path = self.log_dir / "final_training_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_stats = {}
            for key, value in self.training_stats.items():
                if isinstance(value, np.ndarray):
                    json_stats[key] = value.tolist()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    json_stats[key] = [v.tolist() for v in value]
                else:
                    json_stats[key] = value
            json.dump(json_stats, f, indent=2)
        
        # Save best model
        if self.training_stats['best_parameters']:
            best_model_path = self.model_dir / "best_model.pth"
            self.agent.save_model(str(best_model_path))
    
    def plot_training_progress(self, save_path: Optional[str] = None):
        """Plot comprehensive training progress."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Episode rewards
        if self.training_stats['episode_rewards']:
            episodes = range(len(self.training_stats['episode_rewards']))
            axes[0, 0].plot(episodes, self.training_stats['episode_rewards'], alpha=0.6)
            
            # Add moving average
            window = min(50, len(self.training_stats['episode_rewards']) // 10)
            if window > 1:
                moving_avg = np.convolve(self.training_stats['episode_rewards'], 
                                       np.ones(window)/window, mode='valid')
                axes[0, 0].plot(range(window-1, len(self.training_stats['episode_rewards'])), 
                               moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
                axes[0, 0].legend()
            
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].set_title('Training Rewards')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        if self.training_stats['episode_lengths']:
            axes[0, 1].plot(self.training_stats['episode_lengths'])
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Episode Length')
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Success rate
        if self.training_stats['success_rate']:
            axes[0, 2].plot(self.training_stats['success_rate'])
            axes[0, 2].set_xlabel('Episode (x100)')
            axes[0, 2].set_ylabel('Success Rate')
            axes[0, 2].set_title('Success Rate (Rolling 100 episodes)')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Evaluation scores
        if self.training_stats['evaluation_scores']:
            eval_episodes = [e['episode'] for e in self.training_stats['evaluation_scores']]
            eval_scores = [e['score'] for e in self.training_stats['evaluation_scores']]
            axes[1, 0].plot(eval_episodes, eval_scores, 'o-')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Evaluation Score')
            axes[1, 0].set_title('Evaluation Performance')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Noise levels (if available)
        if self.training_stats['noise_levels']:
            axes[1, 1].plot(self.training_stats['noise_levels'])
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Noise Level')
            axes[1, 1].set_title('Noise Level Evolution')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Parameter evolution (show one parameter as example)
        if self.training_stats['parameter_evolution']:
            param_data = self.training_stats['parameter_evolution']
            if param_data and param_data[0]['parameters']:
                param_name = list(param_data[0]['parameters'].keys())[0]
                episodes = [p['episode'] for p in param_data]
                param_values = [p['parameters'][param_name] for p in param_data]
                axes[1, 2].plot(episodes, param_values)
                axes[1, 2].set_xlabel('Episode')
                axes[1, 2].set_ylabel(f'{param_name} Value')
                axes[1, 2].set_title(f'Parameter Evolution: {param_name}')
                axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def test_agent_robustness(self, noise_levels: List[float], 
                            num_episodes_per_level: int = 10) -> Dict:
        """
        Test trained agent robustness to different noise levels.
        
        Args:
            noise_levels: List of noise levels to test
            num_episodes_per_level: Number of episodes per noise level
            
        Returns:
            Dictionary with robustness test results
        """
        print("Testing agent robustness to noise...")
        
        results = {
            'noise_levels': noise_levels,
            'average_rewards': [],
            'success_rates': [],
            'parameter_stability': []
        }
        
        for noise_level in noise_levels:
            print(f"Testing noise level: {noise_level}")
            
            # Create noisy test environment
            test_env = NoisyBioEnvironment(
                oscillator_type=self.env.oscillator_type,
                target_period=self.env.target_period,
                target_amplitude=self.env.target_amplitude,
                noise_strength=noise_level
            )
            
            episode_rewards = []
            success_count = 0
            parameter_variations = []
            
            for _ in range(num_episodes_per_level):
                state = test_env.reset()
                episode_reward = 0
                
                for _ in range(self.max_steps_per_episode):
                    if isinstance(self.agent, DDPG_BioAgent):
                        action = self.agent.act(state, add_noise=False)
                    elif isinstance(self.agent, PPO_BioAgent):
                        action, _, _ = self.agent.act(state)
                    else:
                        action = self.agent.act(state)
                    
                    next_state, reward, done, info = test_env.step(action)
                    episode_reward += reward
                    
                    if done:
                        if info.get('goal_reached', False):
                            success_count += 1
                        break
                    
                    state = next_state
                
                episode_rewards.append(episode_reward)
                
                # Track parameter stability
                if hasattr(test_env, 'get_best_parameters'):
                    best_params = test_env.get_best_parameters()
                    if best_params:
                        parameter_variations.append(best_params)
            
            # Calculate statistics
            avg_reward = np.mean(episode_rewards)
            success_rate = success_count / num_episodes_per_level
            
            # Parameter stability (coefficient of variation)
            param_stability = 0.0
            if parameter_variations:
                param_names = parameter_variations[0].keys()
                stabilities = []
                for param_name in param_names:
                    values = [p[param_name] for p in parameter_variations]
                    cv = np.std(values) / (np.mean(values) + 1e-6)
                    stabilities.append(1.0 / (1.0 + cv))  # Inverse CV as stability
                param_stability = np.mean(stabilities)
            
            results['average_rewards'].append(avg_reward)
            results['success_rates'].append(success_rate)
            results['parameter_stability'].append(param_stability)
        
        return results
    
    def plot_robustness_results(self, robustness_results: Dict, save_path: Optional[str] = None):
        """Plot robustness test results."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        noise_levels = robustness_results['noise_levels']
        
        # Average rewards vs noise
        axes[0].plot(noise_levels, robustness_results['average_rewards'], 'bo-')
        axes[0].set_xlabel('Noise Level')
        axes[0].set_ylabel('Average Reward')
        axes[0].set_title('Reward vs Noise Level')
        axes[0].grid(True, alpha=0.3)
        
        # Success rates vs noise
        axes[1].plot(noise_levels, robustness_results['success_rates'], 'ro-')
        axes[1].set_xlabel('Noise Level')
        axes[1].set_ylabel('Success Rate')
        axes[1].set_title('Success Rate vs Noise Level')
        axes[1].grid(True, alpha=0.3)
        
        # Parameter stability vs noise
        axes[2].plot(noise_levels, robustness_results['parameter_stability'], 'go-')
        axes[2].set_xlabel('Noise Level')
        axes[2].set_ylabel('Parameter Stability')
        axes[2].set_title('Parameter Stability vs Noise Level')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_curriculum_stages(oscillator_type: str = "edelstein") -> List[Dict]:
    """
    Create default curriculum learning stages.
    
    Args:
        oscillator_type: Type of oscillator ("edelstein" or "otero")
        
    Returns:
        List of curriculum stages
    """
    if oscillator_type == "edelstein":
        return [
            {
                'name': 'Easy',
                'episodes': 200,
                'env_params': {
                    'noise_strength': 0.0,
                    'target_period': 10.0,
                    'target_amplitude': 2.0
                }
            },
            {
                'name': 'Medium',
                'episodes': 300,
                'env_params': {
                    'noise_strength': 0.1,
                    'target_period': 8.0,
                    'target_amplitude': 3.0
                }
            },
            {
                'name': 'Hard',
                'episodes': 500,
                'env_params': {
                    'noise_strength': 0.2,
                    'target_period': 12.0,
                    'target_amplitude': 1.5
                }
            }
        ]
    else:  # otero
        return [
            {
                'name': 'Easy',
                'episodes': 200,
                'env_params': {
                    'noise_strength': 0.0,
                    'target_period': 10.0,
                    'target_amplitude': 5.0,
                    'target_phase_coherence': 0.8
                }
            },
            {
                'name': 'Medium',
                'episodes': 300,
                'env_params': {
                    'noise_strength': 0.1,
                    'target_period': 15.0,
                    'target_amplitude': 8.0,
                    'target_phase_coherence': 0.9
                }
            },
            {
                'name': 'Hard',
                'episodes': 500,
                'env_params': {
                    'noise_strength': 0.2,
                    'target_period': 20.0,
                    'target_amplitude': 10.0,
                    'target_phase_coherence': 0.95
                }
            }
        ]
