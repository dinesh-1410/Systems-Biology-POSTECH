"""
Command Line Interface for Edelstein Oscillator

This module provides a command-line interface for running
Edelstein oscillator simulations and optimizations.
"""

import click
import json
import numpy as np
from pathlib import Path
from .oscillator import EdelsteinOscillator
from .optimizer import EdelsteinOptimizer


@click.group()
def main():
    """Edelstein Relaxation Oscillator CLI"""
    pass


@main.command()
@click.option('--k1', default=1.0, help='Rate constant k1')
@click.option('--k2', default=0.5, help='Rate constant k2')
@click.option('--k3', default=2.0, help='Rate constant k3')
@click.option('--k4', default=1.5, help='Rate constant k4')
@click.option('--K1', default=1.0, help='Michaelis constant K1')
@click.option('--K2', default=1.0, help='Michaelis constant K2')
@click.option('--noise', default=0.0, help='Noise strength')
@click.option('--x0', default=1.0, help='Initial X concentration')
@click.option('--y0', default=1.0, help='Initial Y concentration')
@click.option('--tmax', default=50.0, help='Maximum simulation time')
@click.option('--npoints', default=1000, help='Number of time points')
@click.option('--output', '-o', help='Output file path')
@click.option('--plot', is_flag=True, help='Display plot')
def simulate(k1, k2, k3, k4, K1, K2, noise, x0, y0, tmax, npoints, output, plot):
    """Simulate Edelstein oscillator with given parameters."""
    
    # Create oscillator
    oscillator = EdelsteinOscillator(
        k1=k1, k2=k2, k3=k3, k4=k4,
        K1=K1, K2=K2, noise_strength=noise
    )
    
    # Run simulation
    click.echo("Running Edelstein oscillator simulation...")
    result = oscillator.simulate(
        initial_state=(x0, y0),
        time_span=(0, tmax),
        num_points=npoints
    )
    
    # Calculate metrics
    metrics = oscillator.get_fitness_metrics()
    
    # Display results
    click.echo(f"Simulation completed!")
    click.echo(f"Period: {metrics['period']:.4f}" if metrics['period'] else "Period: Not oscillating")
    click.echo(f"Amplitude: {metrics['amplitude']:.4f}" if metrics['amplitude'] else "Amplitude: N/A")
    click.echo(f"Stability: {metrics['stability']:.4f}")
    click.echo(f"Oscillating: {metrics['is_oscillating']}")
    
    # Save results if requested
    if output:
        output_data = {
            'parameters': result['parameters'],
            'time': result['time'].tolist(),
            'X': result['X'].tolist(),
            'Y': result['Y'].tolist(),
            'metrics': metrics
        }
        
        with open(output, 'w') as f:
            json.dump(output_data, f, indent=2)
        click.echo(f"Results saved to {output}")
    
    # Plot if requested
    if plot:
        oscillator.plot_trajectory()


@main.command()
@click.option('--target-period', default=10.0, help='Target oscillation period')
@click.option('--target-amplitude', default=2.0, help='Target oscillation amplitude')
@click.option('--generations', default=100, help='Number of optimization generations')
@click.option('--population', default=50, help='Population size')
@click.option('--mutation-rate', default=0.1, help='Mutation rate')
@click.option('--crossover-rate', default=0.7, help='Crossover rate')
@click.option('--elite-fraction', default=0.2, help='Elite fraction')
@click.option('--output', '-o', help='Output file for best parameters')
@click.option('--plot-progress', is_flag=True, help='Plot optimization progress')
@click.option('--plot-result', is_flag=True, help='Plot final optimized result')
@click.option('--verbose', is_flag=True, help='Verbose output')
def optimize(target_period, target_amplitude, generations, population, 
             mutation_rate, crossover_rate, elite_fraction, output, 
             plot_progress, plot_result, verbose):
    """Optimize Edelstein oscillator parameters using genetic algorithm."""
    
    # Create optimizer
    optimizer = EdelsteinOptimizer(
        target_period=target_period,
        target_amplitude=target_amplitude,
        population_size=population,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        elite_fraction=elite_fraction
    )
    
    # Run optimization
    click.echo("Starting parameter optimization...")
    best_params = optimizer.optimize_parameters(
        generations=generations,
        verbose=verbose
    )
    
    # Validate solution
    validation = optimizer.validate_best_solution()
    
    # Display results
    click.echo("\nOptimization Results:")
    click.echo(f"Best fitness: {validation['fitness']:.4f}")
    click.echo(f"Achieved period: {validation['achieved_period']:.4f} (target: {target_period})")
    click.echo(f"Achieved amplitude: {validation['achieved_amplitude']:.4f} (target: {target_amplitude})")
    click.echo(f"Stability: {validation['stability']:.4f}")
    click.echo(f"Period error: {validation['period_error']:.2%}")
    click.echo(f"Amplitude error: {validation['amplitude_error']:.2%}")
    
    click.echo("\nBest Parameters:")
    for param, value in best_params.items():
        click.echo(f"  {param}: {value:.4f}")
    
    # Save results if requested
    if output:
        output_data = {
            'optimization_settings': {
                'target_period': target_period,
                'target_amplitude': target_amplitude,
                'generations': generations,
                'population_size': population
            },
            'best_parameters': best_params,
            'validation': validation,
            'fitness_history': optimizer.fitness_history
        }
        
        with open(output, 'w') as f:
            json.dump(output_data, f, indent=2)
        click.echo(f"\nResults saved to {output}")
    
    # Plot progress if requested
    if plot_progress:
        optimizer.plot_optimization_progress()
    
    # Plot result if requested
    if plot_result:
        optimizer.plot_best_solution()


@main.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output directory')
@click.option('--plot', is_flag=True, help='Generate plots')
def batch(config_file, output, plot):
    """Run batch simulations from configuration file."""
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    output_dir = Path(output) if output else Path('.')
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    for i, sim_config in enumerate(config.get('simulations', [])):
        click.echo(f"Running simulation {i+1}/{len(config['simulations'])}...")
        
        # Extract parameters
        params = sim_config.get('parameters', {})
        sim_settings = sim_config.get('simulation', {})
        
        # Create and run oscillator
        oscillator = EdelsteinOscillator(**params)
        result = oscillator.simulate(
            initial_state=sim_settings.get('initial_state', (1.0, 1.0)),
            time_span=sim_settings.get('time_span', (0, 50)),
            num_points=sim_settings.get('num_points', 1000)
        )
        
        metrics = oscillator.get_fitness_metrics()
        
        # Store results
        sim_result = {
            'simulation_id': i,
            'parameters': params,
            'metrics': metrics,
            'result': {
                'time': result['time'].tolist(),
                'X': result['X'].tolist(),
                'Y': result['Y'].tolist()
            }
        }
        results.append(sim_result)
        
        # Save individual result
        result_file = output_dir / f"simulation_{i:03d}.json"
        with open(result_file, 'w') as f:
            json.dump(sim_result, f, indent=2)
        
        # Plot if requested
        if plot:
            plot_file = output_dir / f"simulation_{i:03d}.png"
            oscillator.plot_trajectory(save_path=str(plot_file))
    
    # Save summary
    summary_file = output_dir / "batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({'config': config, 'results': results}, f, indent=2)
    
    click.echo(f"Batch simulation completed. Results saved to {output_dir}")


@main.command()
@click.argument('param_file', type=click.Path(exists=True))
@click.option('--noise-levels', default="0.0,0.1,0.2,0.5", help='Comma-separated noise levels')
@click.option('--output', '-o', help='Output directory')
@click.option('--plot', is_flag=True, help='Generate plots')
def noise_analysis(param_file, noise_levels, output, plot):
    """Analyze oscillator behavior under different noise levels."""
    
    with open(param_file, 'r') as f:
        params = json.load(f)
    
    noise_levels = [float(x.strip()) for x in noise_levels.split(',')]
    
    output_dir = Path(output) if output else Path('.')
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    for noise in noise_levels:
        click.echo(f"Analyzing noise level: {noise}")
        
        # Create oscillator with noise
        oscillator = EdelsteinOscillator(noise_strength=noise, **params)
        
        # Run multiple simulations for statistics
        noise_results = []
        for run in range(10):  # 10 runs per noise level
            result = oscillator.simulate(time_span=(0, 100), num_points=2000)
            metrics = oscillator.get_fitness_metrics()
            noise_results.append(metrics)
        
        # Calculate statistics
        periods = [r['period'] for r in noise_results if r['period']]
        amplitudes = [r['amplitude'] for r in noise_results if r['amplitude']]
        stabilities = [r['stability'] for r in noise_results]
        
        stats = {
            'noise_level': noise,
            'period_mean': np.mean(periods) if periods else 0,
            'period_std': np.std(periods) if periods else 0,
            'amplitude_mean': np.mean(amplitudes) if amplitudes else 0,
            'amplitude_std': np.std(amplitudes) if amplitudes else 0,
            'stability_mean': np.mean(stabilities),
            'stability_std': np.std(stabilities),
            'oscillation_rate': sum(r['is_oscillating'] for r in noise_results) / len(noise_results)
        }
        
        results.append(stats)
        
        click.echo(f"  Period: {stats['period_mean']:.3f} ± {stats['period_std']:.3f}")
        click.echo(f"  Amplitude: {stats['amplitude_mean']:.3f} ± {stats['amplitude_std']:.3f}")
        click.echo(f"  Oscillation rate: {stats['oscillation_rate']:.1%}")
    
    # Save results
    results_file = output_dir / "noise_analysis.json"
    with open(results_file, 'w') as f:
        json.dump({'parameters': params, 'results': results}, f, indent=2)
    
    click.echo(f"Noise analysis completed. Results saved to {results_file}")


if __name__ == '__main__':
    main()
