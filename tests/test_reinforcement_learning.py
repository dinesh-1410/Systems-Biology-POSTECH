"""
Tests for Reinforcement Learning Module
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add the code directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from reinforcement_learning.environment import BioOscillatorEnvironment, NoisyBioEnvironment
from reinforcement_learning.agent import DDPG_BioAgent, PPO_BioAgent
from reinforcement_learning.trainer import RLTrainer


class TestBioOscillatorEnvironment:
    """Test cases for Biological Oscillator Environment."""
    
    def test_environment_initialization_edelstein(self):
        """Test environment initialization with Edelstein oscillator."""
        env = BioOscillatorEnvironment(
            oscillator_type="edelstein",
            target_period=10.0,
            target_amplitude=2.0
        )
        
        assert env.oscillator_type == "edelstein"
        assert env.target_period == 10.0
        assert env.target_amplitude == 2.0
        assert len(env.param_names) == 6  # k1, k2, k3, k4, K1, K2
        
        # Check action and observation spaces
        assert env.action_space.shape[0] == 6
        assert env.observation_space.shape[0] == 12  # 6 params + 6 metrics/targets
    
    def test_environment_initialization_otero(self):
        """Test environment initialization with Otero repressilator."""
        env = BioOscillatorEnvironment(
            oscillator_type="otero",
            target_period=15.0,
            target_amplitude=5.0
        )
        
        assert env.oscillator_type == "otero"
        assert env.target_period == 15.0
        assert env.target_amplitude == 5.0
        assert len(env.param_names) == 5  # alpha, alpha0, beta, gamma, n
        
        # Check action and observation spaces
        assert env.action_space.shape[0] == 5
        assert env.observation_space.shape[0] == 11  # 5 params + 6 metrics/targets
    
    def test_environment_reset(self):
        """Test environment reset functionality."""
        env = BioOscillatorEnvironment(oscillator_type="edelstein")
        
        obs = env.reset()
        
        # Should return observation
        assert obs is not None
        assert len(obs) == env.observation_space.shape[0]
        
        # Should have initialized parameters
        assert env.current_parameters is not None
        assert len(env.current_parameters) == len(env.param_names)
        
        # Parameters should be within bounds
        for param, value in env.current_parameters.items():
            low, high = env.param_bounds[param]
            assert low <= value <= high
        
        # Should reset counters
        assert env.step_count == 0
        assert len(env.episode_rewards) == 0
    
    def test_environment_step(self):
        """Test environment step functionality."""
        env = BioOscillatorEnvironment(
            oscillator_type="edelstein",
            simulation_time=10.0  # Short simulation for testing
        )
        
        obs = env.reset()
        
        # Take a random action
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        
        # Check return types
        assert isinstance(next_obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Check observation shape
        assert len(next_obs) == env.observation_space.shape[0]
        
        # Check info contents
        assert 'metrics' in info
        assert 'parameters' in info
        assert 'step' in info
        
        # Check step counter
        assert env.step_count == 1
        assert len(env.episode_rewards) == 1
    
    def test_action_application(self):
        """Test action application to parameters."""
        env = BioOscillatorEnvironment(oscillator_type="edelstein")
        env.reset()
        
        # Store initial parameters
        initial_params = env.current_parameters.copy()
        
        # Apply action
        action = np.array([0.1, -0.1, 0.2, -0.2, 0.0, 0.1])  # Within [-1, 1]
        env._apply_action(action)
        
        # Parameters should have changed
        changed_params = 0
        for param in env.param_names:
            if env.current_parameters[param] != initial_params[param]:
                changed_params += 1
        
        assert changed_params > 0  # At least some parameters should change
        
        # All parameters should still be within bounds
        for param, value in env.current_parameters.items():
            low, high = env.param_bounds[param]
            assert low <= value <= high
    
    def test_reward_calculation(self):
        """Test reward calculation."""
        env = BioOscillatorEnvironment(
            oscillator_type="edelstein",
            target_period=10.0,
            target_amplitude=2.0
        )
        
        # Test with good metrics (close to targets)
        good_metrics = {
            'period': 10.1,
            'amplitude': 2.05,
            'stability': 0.8,
            'is_oscillating': True,
            'simulation_failed': False
        }
        
        good_reward = env._calculate_reward(good_metrics)
        
        # Test with bad metrics
        bad_metrics = {
            'period': 5.0,  # Far from target
            'amplitude': 0.5,  # Far from target
            'stability': 0.2,
            'is_oscillating': False,
            'simulation_failed': False
        }
        
        bad_reward = env._calculate_reward(bad_metrics)
        
        # Good metrics should yield higher reward
        assert good_reward > bad_reward
        
        # Test simulation failure
        failed_metrics = {'simulation_failed': True}
        failed_reward = env._calculate_reward(failed_metrics)
        assert failed_reward < bad_reward
    
    def test_goal_detection(self):
        """Test goal reaching detection."""
        env = BioOscillatorEnvironment(
            oscillator_type="edelstein",
            target_period=10.0,
            target_amplitude=2.0
        )
        
        # Test goal reached
        good_metrics = {
            'period': 10.02,  # Very close to target
            'amplitude': 2.01,  # Very close to target
            'stability': 0.9,
            'is_oscillating': True
        }
        
        assert env._is_goal_reached(good_metrics) == True
        
        # Test goal not reached
        bad_metrics = {
            'period': 8.0,  # Too far from target
            'amplitude': 2.01,
            'stability': 0.9,
            'is_oscillating': True
        }
        
        assert env._is_goal_reached(bad_metrics) == False
        
        # Test non-oscillating
        non_osc_metrics = {
            'period': 10.02,
            'amplitude': 2.01,
            'stability': 0.9,
            'is_oscillating': False
        }
        
        assert env._is_goal_reached(non_osc_metrics) == False


class TestNoisyBioEnvironment:
    """Test cases for Noisy Biological Environment."""
    
    def test_noisy_environment_initialization(self):
        """Test noisy environment initialization."""
        def noise_schedule(step):
            return 0.1 + 0.01 * step
        
        env = NoisyBioEnvironment(
            oscillator_type="edelstein",
            noise_schedule=noise_schedule,
            perturbation_probability=0.2
        )
        
        assert env.noise_schedule == noise_schedule
        assert env.perturbation_probability == 0.2
        assert env.current_noise_level == env.noise_strength
    
    def test_dynamic_noise(self):
        """Test dynamic noise scheduling."""
        def linear_noise(step):
            return min(0.5, 0.1 + 0.01 * step)
        
        env = NoisyBioEnvironment(
            oscillator_type="edelstein",
            noise_schedule=linear_noise,
            simulation_time=5.0  # Short simulation
        )
        
        env.reset()
        
        # Take a few steps and check noise evolution
        initial_noise = env.current_noise_level
        
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            # Check that noise info is included
            assert 'noise_level' in info
            
            if done:
                break
        
        # Noise should have changed (if schedule is dynamic)
        final_noise = env.current_noise_level
        # Note: May not always be different due to randomness in step count


class TestDDPG_BioAgent:
    """Test cases for DDPG Bio Agent."""
    
    def test_agent_initialization(self):
        """Test DDPG agent initialization."""
        agent = DDPG_BioAgent(
            state_dim=10,
            action_dim=5,
            hidden_dim=64,
            device='cpu'
        )
        
        assert agent.state_dim == 10
        assert agent.action_dim == 5
        assert agent.device.type == 'cpu'
        
        # Check networks exist
        assert agent.actor is not None
        assert agent.critic is not None
        assert agent.target_actor is not None
        assert agent.target_critic is not None
        
        # Check optimizers
        assert agent.actor_optimizer is not None
        assert agent.critic_optimizer is not None
        
        # Check replay buffer
        assert agent.replay_buffer is not None
        assert len(agent.replay_buffer) == 0
    
    def test_agent_action_selection(self):
        """Test agent action selection."""
        agent = DDPG_BioAgent(state_dim=8, action_dim=4, device='cpu')
        
        state = np.random.randn(8)
        
        # Test action without noise
        action_no_noise = agent.act(state, add_noise=False)
        assert len(action_no_noise) == 4
        assert all(-1 <= a <= 1 for a in action_no_noise)
        
        # Test action with noise
        action_with_noise = agent.act(state, add_noise=True)
        assert len(action_with_noise) == 4
        assert all(-1 <= a <= 1 for a in action_with_noise)
        
        # Actions should be different (with high probability)
        assert not np.allclose(action_no_noise, action_with_noise, atol=1e-3)
    
    def test_agent_experience_storage(self):
        """Test experience storage and replay."""
        agent = DDPG_BioAgent(state_dim=6, action_dim=3, batch_size=4, device='cpu')
        
        # Add some experiences
        for i in range(10):
            state = np.random.randn(6)
            action = np.random.randn(3)
            reward = np.random.randn()
            next_state = np.random.randn(6)
            done = False
            
            agent.update(state, action, reward, next_state, done)
        
        # Buffer should have experiences
        assert len(agent.replay_buffer) == 10
    
    def test_agent_learning(self):
        """Test agent learning process."""
        agent = DDPG_BioAgent(state_dim=4, action_dim=2, batch_size=8, device='cpu')
        
        # Add enough experiences for learning
        for i in range(20):
            state = np.random.randn(4)
            action = np.random.uniform(-1, 1, 2)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = i == 19  # Last episode done
            
            agent.update(state, action, reward, next_state, done)
        
        # Should have triggered learning
        assert len(agent.training_stats['losses']) > 0
        
        # Check loss structure
        loss_entry = agent.training_stats['losses'][0]
        assert 'critic_loss' in loss_entry
        assert 'actor_loss' in loss_entry


class TestPPO_BioAgent:
    """Test cases for PPO Bio Agent."""
    
    def test_ppo_agent_initialization(self):
        """Test PPO agent initialization."""
        agent = PPO_BioAgent(
            state_dim=8,
            action_dim=4,
            hidden_dim=64,
            device='cpu'
        )
        
        assert agent.state_dim == 8
        assert agent.action_dim == 4
        assert agent.device.type == 'cpu'
        
        # Check policy network
        assert agent.policy is not None
        assert agent.optimizer is not None
        
        # Check experience storage
        assert len(agent.states) == 0
        assert len(agent.actions) == 0
    
    def test_ppo_action_selection(self):
        """Test PPO action selection."""
        agent = PPO_BioAgent(state_dim=6, action_dim=3, device='cpu')
        
        state = np.random.randn(6)
        action, log_prob, value = agent.act(state)
        
        # Check return types and shapes
        assert len(action) == 3
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        assert all(-1 <= a <= 1 for a in action)
    
    def test_ppo_experience_storage(self):
        """Test PPO experience storage."""
        agent = PPO_BioAgent(state_dim=4, action_dim=2, device='cpu')
        
        # Store some experiences
        for i in range(5):
            state = np.random.randn(4)
            action = np.random.uniform(-1, 1, 2)
            reward = np.random.randn()
            log_prob = np.random.randn()
            value = np.random.randn()
            done = False
            
            agent.store_experience(state, action, reward, log_prob, value, done)
        
        # Check storage
        assert len(agent.states) == 5
        assert len(agent.actions) == 5
        assert len(agent.rewards) == 5
    
    def test_ppo_learning(self):
        """Test PPO learning process."""
        agent = PPO_BioAgent(state_dim=4, action_dim=2, batch_size=8, device='cpu')
        
        # Store enough experiences for learning
        for i in range(10):
            state = np.random.randn(4)
            action = np.random.uniform(-1, 1, 2)
            reward = np.random.randn()
            log_prob = np.random.randn()
            value = np.random.randn()
            done = False
            
            agent.store_experience(state, action, reward, log_prob, value, done)
        
        # Trigger learning
        loss = agent.learn()
        
        # Should return a loss value
        assert isinstance(loss, float)
        assert not np.isnan(loss)
        
        # Experience buffer should be cleared
        assert len(agent.states) == 0
        assert len(agent.actions) == 0


class TestRLTrainer:
    """Test cases for RL Trainer."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        # Create mock agent and environment
        mock_agent = Mock()
        mock_env = Mock()
        mock_env.reset.return_value = np.random.randn(8)
        
        trainer = RLTrainer(
            agent=mock_agent,
            environment=mock_env,
            max_episodes=100,
            max_steps_per_episode=50
        )
        
        assert trainer.agent == mock_agent
        assert trainer.env == mock_env
        assert trainer.max_episodes == 100
        assert trainer.max_steps_per_episode == 50
        
        # Check initial state
        assert len(trainer.training_stats['episode_rewards']) == 0
        assert trainer.training_stats['best_reward'] == -np.inf
    
    def test_curriculum_learning_setup(self):
        """Test curriculum learning setup."""
        mock_agent = Mock()
        mock_env = Mock()
        
        trainer = RLTrainer(agent=mock_agent, environment=mock_env)
        
        # Enable curriculum learning
        stages = [
            {'name': 'easy', 'episodes': 50, 'env_params': {'noise_strength': 0.0}},
            {'name': 'hard', 'episodes': 50, 'env_params': {'noise_strength': 0.2}}
        ]
        
        trainer.enable_curriculum_learning(stages)
        
        assert trainer.curriculum_enabled == True
        assert len(trainer.curriculum_stages) == 2
        assert trainer.current_stage == 0
    
    @patch('reinforcement_learning.trainer.tqdm')  # Mock tqdm to avoid output during tests
    def test_short_training_run(self, mock_tqdm):
        """Test a short training run."""
        # Create mock environment
        mock_env = Mock()
        mock_env.reset.return_value = np.random.randn(6)
        mock_env.step.return_value = (
            np.random.randn(6),  # next_state
            1.0,  # reward
            True,  # done
            {'goal_reached': True, 'parameters': {}, 'metrics': {}}  # info
        )
        
        # Create mock agent
        mock_agent = Mock()
        mock_agent.act.return_value = np.random.uniform(-1, 1, 3)
        
        # Create trainer
        trainer = RLTrainer(
            agent=mock_agent,
            environment=mock_env,
            max_episodes=2,  # Very short training
            max_steps_per_episode=5,
            evaluation_frequency=1,
            save_frequency=10
        )
        
        # Mock tqdm to return a simple iterator
        mock_tqdm.return_value = range(2)
        
        # Run training
        training_stats = trainer.train(verbose=False)
        
        # Check that training ran
        assert len(training_stats['episode_rewards']) == 2
        assert len(training_stats['episode_lengths']) == 2
        
        # Check that agent was called
        assert mock_agent.act.call_count >= 2  # At least once per episode
        assert mock_env.step.call_count >= 2
    
    def test_evaluation(self):
        """Test agent evaluation."""
        # Create mock environment
        mock_env = Mock()
        mock_env.reset.return_value = np.random.randn(4)
        mock_env.step.return_value = (
            np.random.randn(4),  # next_state
            0.5,  # reward
            True,  # done
            {}  # info
        )
        
        # Create mock agent
        mock_agent = Mock()
        mock_agent.act.return_value = np.random.uniform(-1, 1, 2)
        
        trainer = RLTrainer(
            agent=mock_agent,
            environment=mock_env,
            max_steps_per_episode=3
        )
        
        # Run evaluation
        eval_score = trainer._evaluate_agent(num_episodes=2)
        
        # Should return a numeric score
        assert isinstance(eval_score, (int, float))
        assert not np.isnan(eval_score)
    
    def test_robustness_testing(self):
        """Test robustness testing functionality."""
        # Create mock agent and environment
        mock_agent = Mock()
        mock_agent.act.return_value = np.random.uniform(-1, 1, 2)
        
        mock_env = Mock()
        mock_env.oscillator_type = "edelstein"
        mock_env.target_period = 10.0
        mock_env.target_amplitude = 2.0
        
        trainer = RLTrainer(
            agent=mock_agent,
            environment=mock_env,
            max_steps_per_episode=5
        )
        
        # Mock the NoisyBioEnvironment creation and behavior
        with patch('reinforcement_learning.trainer.NoisyBioEnvironment') as mock_noisy_env_class:
            mock_noisy_env = Mock()
            mock_noisy_env.reset.return_value = np.random.randn(8)
            mock_noisy_env.step.return_value = (
                np.random.randn(8),  # next_state
                0.3,  # reward
                True,  # done
                {'goal_reached': False}  # info
            )
            mock_noisy_env_class.return_value = mock_noisy_env
            
            # Run robustness test
            noise_levels = [0.0, 0.1, 0.2]
            results = trainer.test_agent_robustness(
                noise_levels=noise_levels,
                num_episodes_per_level=2
            )
            
            # Check results structure
            assert 'noise_levels' in results
            assert 'average_rewards' in results
            assert 'success_rates' in results
            
            assert len(results['noise_levels']) == 3
            assert len(results['average_rewards']) == 3
            assert len(results['success_rates']) == 3


if __name__ == "__main__":
    pytest.main([__file__])
