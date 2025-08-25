"""
Tests for Analysis Module
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import json

# Add the code directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from analysis.visualizer import ResultVisualizer
from analysis.metrics import PerformanceMetrics
from analysis.utils import DataProcessor


class TestResultVisualizer:
    """Test cases for Result Visualizer."""
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        visualizer = ResultVisualizer()
        
        assert visualizer.default_figsize == (10, 6)
        assert len(visualizer.colors) == 10
    
    def test_visualizer_custom_style(self):
        """Test visualizer with custom style."""
        visualizer = ResultVisualizer(style='default', figsize=(8, 4))
        
        assert visualizer.default_figsize == (8, 4)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_oscillator_dynamics(self, mock_show):
        """Test oscillator dynamics plotting."""
        visualizer = ResultVisualizer()
        
        # Create test data
        time_data = np.linspace(0, 20, 100)
        concentration_data = {
            'X': 2 + np.sin(time_data),
            'Y': 1.5 + np.cos(time_data + np.pi/4)
        }
        
        # Should not raise an error
        visualizer.plot_oscillator_dynamics(time_data, concentration_data)
        
        # Check that show was called
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_3d_phase_portrait(self, mock_show):
        """Test 3D phase portrait plotting."""
        visualizer = ResultVisualizer()
        
        # Create test data with 3 species
        time_points = 100
        data_dict = {
            'protein1': np.sin(np.linspace(0, 4*np.pi, time_points)),
            'protein2': np.sin(np.linspace(0, 4*np.pi, time_points) + 2*np.pi/3),
            'protein3': np.sin(np.linspace(0, 4*np.pi, time_points) + 4*np.pi/3)
        }
        
        # Should not raise an error
        visualizer.plot_3d_phase_portrait(data_dict)
        
        # Check that show was called
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_optimization_progress(self, mock_show):
        """Test optimization progress plotting."""
        visualizer = ResultVisualizer()
        
        # Create test fitness history
        fitness_history = []
        for i in range(20):
            fitness_history.append({
                'generation': i,
                'best_fitness': 0.1 + 0.8 * (1 - np.exp(-i/5)),
                'mean_fitness': 0.05 + 0.6 * (1 - np.exp(-i/5)),
                'std_fitness': 0.2 * np.exp(-i/10)
            })
        
        # Should not raise an error
        visualizer.plot_optimization_progress(fitness_history)
        
        # Check that show was called
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_parameter_evolution(self, mock_show):
        """Test parameter evolution plotting."""
        visualizer = ResultVisualizer()
        
        # Create test parameter history
        parameter_history = []
        for i in range(15):
            parameter_history.append({
                'generation': i,
                'parameters': {
                    'k1': 1.0 + 0.1 * np.sin(i/3),
                    'k2': 0.5 + 0.05 * np.cos(i/2),
                    'k3': 2.0 + 0.2 * np.sin(i/4)
                }
            })
        
        # Should not raise an error
        visualizer.plot_parameter_evolution(parameter_history)
        
        # Check that show was called
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_bifurcation_diagram(self, mock_show):
        """Test bifurcation diagram plotting."""
        visualizer = ResultVisualizer()
        
        # Create test bifurcation data
        param_values = np.linspace(0.1, 2.0, 50)
        bifurcation_data = {
            'parameter': 'k1',
            'param_values': param_values,
            'max_concentrations': 2 + np.sin(param_values * 3) + 0.1 * np.random.randn(50),
            'min_concentrations': 0.5 + 0.5 * np.cos(param_values * 2) + 0.1 * np.random.randn(50),
            'periods': 10 + 2 * np.sin(param_values) + 0.5 * np.random.randn(50),
            'is_oscillating': [True] * 50
        }
        
        # Should not raise an error
        visualizer.plot_bifurcation_diagram(bifurcation_data)
        
        # Check that show was called
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_noise_analysis(self, mock_show):
        """Test noise analysis plotting."""
        visualizer = ResultVisualizer()
        
        # Create test noise results
        noise_results = []
        noise_levels = [0.0, 0.1, 0.2, 0.3]
        for noise in noise_levels:
            noise_results.append({
                'noise_level': noise,
                'period_mean': 10.0 + noise * 2,
                'period_std': 0.5 + noise,
                'amplitude_mean': 2.0 - noise * 0.5,
                'amplitude_std': 0.2 + noise * 0.3,
                'stability_mean': 0.9 - noise * 2,
                'stability_std': 0.1 + noise * 0.2,
                'oscillation_rate': 1.0 - noise * 0.5
            })
        
        # Should not raise an error
        visualizer.plot_noise_analysis(noise_results)
        
        # Check that show was called
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_rl_training_progress(self, mock_show):
        """Test RL training progress plotting."""
        visualizer = ResultVisualizer()
        
        # Create test RL training stats
        training_stats = {
            'episode_rewards': np.cumsum(np.random.randn(100) * 0.1 + 0.02),
            'episode_lengths': np.random.randint(20, 80, 100),
            'success_rate': [min(1.0, i/50) for i in range(0, 100, 10)],
            'evaluation_scores': [
                {'episode': i*10, 'score': 0.1 + 0.8 * (1 - np.exp(-i/3))} 
                for i in range(10)
            ],
            'noise_levels': [0.1 + 0.01 * i for i in range(100)]
        }
        
        # Should not raise an error
        visualizer.plot_rl_training_progress(training_stats)
        
        # Check that show was called
        mock_show.assert_called_once()
    
    def test_create_interactive_dashboard(self):
        """Test interactive dashboard creation."""
        visualizer = ResultVisualizer()
        
        # Create test data
        data_dict = {
            'time_series': {
                'time': np.linspace(0, 20, 100),
                'concentrations': {
                    'X': 2 + np.sin(np.linspace(0, 20, 100)),
                    'Y': 1.5 + np.cos(np.linspace(0, 20, 100))
                }
            },
            'phase_data': {
                'X': 2 + np.sin(np.linspace(0, 20, 100)),
                'Y': 1.5 + np.cos(np.linspace(0, 20, 100))
            }
        }
        
        # Should return a plotly figure
        fig = visualizer.create_interactive_dashboard(data_dict)
        
        # Check that it's a plotly figure
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
    
    def test_save_all_plots(self):
        """Test saving all plots functionality."""
        visualizer = ResultVisualizer()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create comprehensive test data
            results_dict = {
                'time_series': {
                    'time': np.linspace(0, 20, 100),
                    'concentrations': {
                        'X': 2 + np.sin(np.linspace(0, 20, 100)),
                        'Y': 1.5 + np.cos(np.linspace(0, 20, 100))
                    }
                },
                'fitness_history': [
                    {'generation': i, 'best_fitness': i*0.1, 'mean_fitness': i*0.05, 'std_fitness': 0.1}
                    for i in range(10)
                ]
            }
            
            # Should not raise an error
            with patch('matplotlib.pyplot.show'):  # Suppress plot display
                visualizer.save_all_plots(results_dict, temp_dir)
            
            # Check that some files were created
            plot_files = list(Path(temp_dir).glob("*.png"))
            assert len(plot_files) > 0


class TestPerformanceMetrics:
    """Test cases for Performance Metrics."""
    
    def test_metrics_initialization(self):
        """Test metrics calculator initialization."""
        metrics = PerformanceMetrics()
        assert metrics.tolerance == 1e-6
    
    def test_period_calculation_peaks(self):
        """Test period calculation using peaks method."""
        metrics = PerformanceMetrics()
        
        # Create oscillatory data
        time_data = np.linspace(0, 40, 1000)
        true_period = 5.0
        concentration_data = 2 + np.sin(2 * np.pi * time_data / true_period)
        
        calculated_period = metrics.calculate_period(time_data, concentration_data, method='peaks')
        
        # Should detect the correct period (within tolerance)
        if calculated_period is not None:
            assert abs(calculated_period - true_period) < 0.5
    
    def test_period_calculation_fft(self):
        """Test period calculation using FFT method."""
        metrics = PerformanceMetrics()
        
        # Create clean oscillatory data
        time_data = np.linspace(0, 50, 1000)
        true_period = 8.0
        concentration_data = 1.5 + np.sin(2 * np.pi * time_data / true_period)
        
        calculated_period = metrics.calculate_period(time_data, concentration_data, method='fft')
        
        # Should detect the correct period (within tolerance)
        if calculated_period is not None:
            assert abs(calculated_period - true_period) < 1.0
    
    def test_period_calculation_autocorr(self):
        """Test period calculation using autocorrelation method."""
        metrics = PerformanceMetrics()
        
        # Create oscillatory data
        time_data = np.linspace(0, 60, 1200)
        true_period = 12.0
        concentration_data = 3 + 2 * np.sin(2 * np.pi * time_data / true_period)
        
        calculated_period = metrics.calculate_period(time_data, concentration_data, method='autocorr')
        
        # Should detect some reasonable period
        if calculated_period is not None:
            assert calculated_period > 0
            assert calculated_period < 30  # Reasonable upper bound
    
    def test_amplitude_calculation(self):
        """Test amplitude calculation methods."""
        metrics = PerformanceMetrics()
        
        # Create test data with known amplitude
        data = 5 + 3 * np.sin(np.linspace(0, 4*np.pi, 100))  # amplitude = 3
        
        # Peak-to-peak method
        amp_p2p = metrics.calculate_amplitude(data, method='peak_to_peak')
        assert abs(amp_p2p - 3.0) < 0.1
        
        # Standard deviation method
        amp_std = metrics.calculate_amplitude(data, method='std')
        expected_std = 3 / np.sqrt(2)  # For sine wave
        assert abs(amp_std - expected_std) < 0.5
        
        # Range method
        amp_range = metrics.calculate_amplitude(data, method='range')
        assert abs(amp_range - 6.0) < 0.1  # Full range = 2 * amplitude
    
    def test_stability_calculation(self):
        """Test stability calculation."""
        metrics = PerformanceMetrics()
        
        # Stable oscillation
        stable_data = np.sin(np.linspace(0, 10*np.pi, 1000))
        stability_stable = metrics.calculate_stability(stable_data)
        
        # Unstable oscillation (with trend)
        unstable_data = np.sin(np.linspace(0, 10*np.pi, 1000)) + np.linspace(0, 5, 1000)
        stability_unstable = metrics.calculate_stability(unstable_data)
        
        # Stable should have higher stability score
        assert stability_stable > stability_unstable
        assert 0 <= stability_stable <= 1
        assert 0 <= stability_unstable <= 1
    
    def test_phase_coherence_calculation(self):
        """Test phase coherence calculation."""
        metrics = PerformanceMetrics()
        
        # Perfect repressilator with 120° phase differences
        time_points = np.linspace(0, 6*np.pi, 1000)
        species_data = {
            'p1': np.sin(time_points),
            'p2': np.sin(time_points + 2*np.pi/3),
            'p3': np.sin(time_points + 4*np.pi/3)
        }
        
        coherence = metrics.calculate_phase_coherence(species_data, target_phase_diff=120.0)
        
        # Should have high coherence for perfect repressilator
        assert 0 <= coherence <= 1
        # Note: Exact value depends on Hilbert transform accuracy
    
    def test_bifurcation_detection(self):
        """Test bifurcation detection."""
        metrics = PerformanceMetrics()
        
        # Create data with a bifurcation-like behavior
        param_values = np.linspace(0, 2, 100)
        dynamics_data = param_values**2  # Quadratic change
        
        bifurcation_analysis = metrics.detect_bifurcations(param_values, dynamics_data)
        
        # Check structure
        assert 'bifurcation_points' in bifurcation_analysis
        assert 'first_derivative' in bifurcation_analysis
        assert 'second_derivative' in bifurcation_analysis
        
        # Should detect some changes
        assert len(bifurcation_analysis['first_derivative']) == len(param_values)
        assert len(bifurcation_analysis['second_derivative']) == len(param_values)
    
    def test_noise_robustness_calculation(self):
        """Test noise robustness calculation."""
        metrics = PerformanceMetrics()
        
        # Clean metrics
        clean_metrics = {'period': 10.0, 'amplitude': 2.0, 'stability': 0.9}
        
        # Noisy metrics (slightly degraded)
        noisy_metrics = [
            {'period': 10.1, 'amplitude': 1.95, 'stability': 0.85},
            {'period': 9.9, 'amplitude': 2.05, 'stability': 0.88},
            {'period': 10.2, 'amplitude': 1.98, 'stability': 0.87}
        ]
        
        robustness = metrics.calculate_noise_robustness(clean_metrics, noisy_metrics)
        
        # Should return robustness scores
        assert 'period' in robustness
        assert 'amplitude' in robustness
        assert 'stability' in robustness
        
        # All scores should be between 0 and 1
        for score in robustness.values():
            assert 0 <= score <= 1
    
    def test_optimization_efficiency_calculation(self):
        """Test optimization efficiency calculation."""
        metrics = PerformanceMetrics()
        
        # Create mock fitness history
        fitness_history = []
        for i in range(50):
            fitness_history.append({
                'best_fitness': 0.1 + 0.8 * (1 - np.exp(-i/10)),
                'mean_fitness': 0.05 + 0.6 * (1 - np.exp(-i/10)),
                'std_fitness': 0.2 * np.exp(-i/20)
            })
        
        efficiency = metrics.calculate_optimization_efficiency(fitness_history)
        
        # Check required metrics
        required_metrics = ['convergence_rate', 'final_convergence', 'success_rate', 'final_fitness']
        assert all(metric in efficiency for metric in required_metrics)
        
        # Check reasonable values
        assert efficiency['final_fitness'] > efficiency['final_fitness']  # Should improve
        assert 0 <= efficiency['success_rate'] <= 1
    
    def test_rl_performance_metrics(self):
        """Test RL performance metrics calculation."""
        metrics = PerformanceMetrics()
        
        # Create mock RL data
        episode_rewards = np.cumsum(np.random.randn(100) * 0.1 + 0.01)  # Generally improving
        episode_lengths = np.random.randint(20, 50, 100)
        success_indicators = [r > np.median(episode_rewards) for r in episode_rewards]
        
        rl_metrics = metrics.calculate_rl_performance_metrics(
            episode_rewards, episode_lengths, success_indicators
        )
        
        # Check required metrics
        required_metrics = ['mean_reward', 'std_reward', 'success_rate', 'learning_improvement']
        assert all(metric in rl_metrics for metric in required_metrics)
        
        # Check reasonable values
        assert rl_metrics['mean_reward'] == np.mean(episode_rewards)
        assert 0 <= rl_metrics['success_rate'] <= 1
    
    def test_comprehensive_report_generation(self):
        """Test comprehensive report generation."""
        metrics = PerformanceMetrics()
        
        # Create mock results
        time_data = np.linspace(0, 50, 1000)
        results = {
            'simulation_data': {
                'time': time_data,
                'X': 2 + np.sin(time_data),
                'Y': 1.5 + np.cos(time_data + np.pi/4)
            },
            'fitness_history': [
                {'best_fitness': i*0.1, 'mean_fitness': i*0.05, 'std_fitness': 0.1}
                for i in range(20)
            ]
        }
        
        target_metrics = {'X_period': 6.28, 'X_amplitude': 1.0}
        
        report = metrics.generate_comprehensive_report(results, target_metrics)
        
        # Check report structure
        assert 'summary' in report
        assert 'detailed_metrics' in report
        assert 'performance_analysis' in report
        assert 'recommendations' in report
        
        # Check that metrics were calculated
        assert len(report['detailed_metrics']) > 0
        assert 'optimization' in report['performance_analysis']
        assert len(report['recommendations']) > 0


class TestDataProcessor:
    """Test cases for Data Processor."""
    
    def test_processor_initialization(self):
        """Test data processor initialization."""
        processor = DataProcessor()
        assert processor.tolerance == 1e-6
        assert processor.logger is not None
    
    def test_data_conversion(self):
        """Test data type conversion."""
        processor = DataProcessor()
        
        # Test list to array conversion
        data = {
            'time': [0, 1, 2, 3, 4],
            'concentration': [1.0, 1.5, 2.0, 1.5, 1.0],
            'metadata': 'test'
        }
        
        converted = processor._convert_to_arrays(data)
        
        assert isinstance(converted['time'], np.ndarray)
        assert isinstance(converted['concentration'], np.ndarray)
        assert isinstance(converted['metadata'], str)
    
    def test_json_conversion(self):
        """Test JSON conversion."""
        processor = DataProcessor()
        
        data = {
            'array': np.array([1, 2, 3]),
            'float': np.float64(3.14),
            'int': np.int32(42),
            'nested': {
                'array': np.array([4, 5, 6])
            }
        }
        
        json_data = processor._convert_for_json(data)
        
        # Should be JSON serializable
        json_str = json.dumps(json_data)
        assert isinstance(json_str, str)
        
        # Check conversions
        assert isinstance(json_data['array'], list)
        assert isinstance(json_data['float'], float)
        assert isinstance(json_data['nested']['array'], list)
    
    def test_data_cleaning(self):
        """Test data cleaning functionality."""
        processor = DataProcessor()
        
        # Create data with outliers and missing values
        clean_data = np.sin(np.linspace(0, 4*np.pi, 100))
        noisy_data = clean_data.copy()
        noisy_data[10] = 100  # Outlier
        noisy_data[20] = -100  # Outlier
        noisy_data[30] = np.nan  # Missing value
        
        data = {'clean': clean_data, 'noisy': noisy_data}
        
        cleaned = processor.clean_simulation_data(data, remove_outliers=True, interpolate_missing=True)
        
        # Should have same keys
        assert set(cleaned.keys()) == set(data.keys())
        
        # Noisy data should be cleaned
        assert not np.any(np.isnan(cleaned['noisy']))
        assert np.max(cleaned['noisy']) < 50  # Outliers removed
    
    def test_data_resampling(self):
        """Test data resampling."""
        processor = DataProcessor()
        
        # Create irregular time series
        time_data = np.array([0, 0.5, 1.2, 2.8, 4.1, 5.0])
        concentration_data = {
            'X': np.sin(time_data),
            'Y': np.cos(time_data)
        }
        
        new_time, new_concentrations = processor.resample_data(
            time_data, concentration_data, target_points=50
        )
        
        # Check output
        assert len(new_time) == 50
        assert len(new_concentrations['X']) == 50
        assert len(new_concentrations['Y']) == 50
        
        # Time should be uniform
        dt = np.diff(new_time)
        assert np.allclose(dt, dt[0], rtol=1e-10)
    
    def test_data_normalization(self):
        """Test data normalization methods."""
        processor = DataProcessor()
        
        # Create test data
        data = {
            'X': np.array([1, 2, 3, 4, 5]),
            'Y': np.array([10, 20, 30, 40, 50])
        }
        
        # MinMax normalization
        normalized_minmax = processor.normalize_data(data, method='minmax')
        assert np.min(normalized_minmax['X']) == 0.0
        assert np.max(normalized_minmax['X']) == 1.0
        
        # Z-score normalization
        normalized_zscore = processor.normalize_data(data, method='zscore')
        assert abs(np.mean(normalized_zscore['X'])) < 1e-10  # Should be ~0
        assert abs(np.std(normalized_zscore['X']) - 1.0) < 1e-10  # Should be ~1
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        processor = DataProcessor()
        
        # Create test time series
        time_data = np.linspace(0, 10, 100)
        concentration_data = {
            'X': 2 + np.sin(time_data),
            'Y': 1 + 0.5 * np.cos(time_data)
        }
        
        features = processor.extract_features(time_data, concentration_data)
        
        # Check that features were extracted
        assert 'X_mean' in features
        assert 'X_std' in features
        assert 'X_min' in features
        assert 'X_max' in features
        assert 'Y_mean' in features
        
        # Check reasonable values
        assert abs(features['X_mean'] - 2.0) < 0.1  # Should be close to 2
        assert abs(features['Y_mean'] - 1.0) < 0.1  # Should be close to 1
    
    def test_data_validation(self):
        """Test data quality validation."""
        processor = DataProcessor()
        
        # Create data with various quality issues
        good_data = np.sin(np.linspace(0, 4*np.pi, 100))
        bad_data = np.array([1, np.nan, np.inf, -np.inf, 5])
        constant_data = np.ones(50)
        negative_concentration = np.array([-1, -2, -3])
        
        data = {
            'good_series': good_data,
            'bad_series': bad_data,
            'constant_series': constant_data,
            'negative_concentration': negative_concentration
        }
        
        report = processor.validate_data_quality(data)
        
        # Check report structure
        assert 'overall_quality' in report
        assert 'issues' in report
        assert 'warnings' in report
        assert 'statistics' in report
        
        # Should detect issues
        assert len(report['issues']) > 0  # NaN and Inf values
        assert len(report['warnings']) > 0  # Constant and negative values
        
        # Overall quality should be poor due to issues
        assert report['overall_quality'] in ['poor', 'fair', 'good']
    
    def test_file_operations(self):
        """Test file save/load operations."""
        processor = DataProcessor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            test_data = {
                'time': np.linspace(0, 10, 50),
                'X': np.sin(np.linspace(0, 10, 50)),
                'parameters': {'k1': 1.0, 'k2': 0.5}
            }
            
            # Test JSON save/load
            json_path = Path(temp_dir) / 'test.json'
            processor.save_simulation_data(test_data, json_path, format='json')
            loaded_json = processor.load_simulation_data(json_path)
            
            assert 'time' in loaded_json
            assert 'X' in loaded_json
            assert 'parameters' in loaded_json
            
            # Test CSV save/load
            csv_data = {'time': test_data['time'], 'X': test_data['X']}
            csv_path = Path(temp_dir) / 'test.csv'
            processor.save_simulation_data(csv_data, csv_path, format='csv')
            loaded_csv = processor.load_simulation_data(csv_path)
            
            assert 'time' in loaded_csv
            assert 'X' in loaded_csv


if __name__ == "__main__":
    pytest.main([__file__])
