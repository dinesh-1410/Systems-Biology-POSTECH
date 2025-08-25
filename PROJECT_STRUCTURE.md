# Project Structure

This document provides a comprehensive overview of the Systems Biology project repository structure for optimizing synthetic oscillatory biological networks through Reinforcement Learning.

## Repository Overview

```
SYS-BIO/
├── README.md                           # Main project documentation
├── PROJECT_STRUCTURE.md                # This file - detailed structure
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore patterns
│
├── code/                              # Main implementation code
│   ├── README.md                      # Code documentation
│   │
│   ├── edelstein_oscillator/          # Edelstein Relaxation Oscillator
│   │   ├── __init__.py                # Package initialization
│   │   ├── oscillator.py              # Core oscillator model 
│   │   ├── optimizer.py               # RL-based optimizer 
│   │   └── cli.py                     # Command-line interface 
│   │
│   ├── otero_repressilator/           # Otero Repressilator (3-gene circuit)
│   │   ├── __init__.py                # Package initialization
│   │   ├── repressilator.py           # Core repressilator mode 
│   │   ├── optimizer.py               # RL-based optimizer 
│   │   └── cli.py                     # Command-line interface 
│   │
│   ├── reinforcement_learning/        # RL algorithms and training
│   │   ├── __init__.py                # Package initialization
│   │   ├── agent.py                   # DDPG & PPO agents 
│   │   ├── environment.py             # Biological environments 
│   │   ├── trainer.py                 # Training framework 
│   │   └── cli.py                     # Command-line interface 
│   │
│   └── analysis/                      # Analysis and visualization tools
│       ├── __init__.py                # Package initialization
│       ├── visualizer.py              # Result visualization 
│       ├── metrics.py                 # Performance metrics 
│       └── utils.py                   # Utility functions 
│
├── tests/                             # Test suite
│   ├── __init__.py                    # Test package initialization
│   ├── test_edelstein_oscillator.py   # Edelstein oscillator tests 
│   ├── test_otero_repressilator.py    # Otero repressilator tests 
│   ├── test_reinforcement_learning.py # RL module tests 
│   └── test_analysis.py               # Analysis module tests 
│
├── .github/                           # GitHub configuration
│   └── workflows/                     # CI/CD workflows
│       └── ci.yml                     # Continuous integration 
│
├── Presentation/                      # Course presentation materials
│   ├── Dinesh Saggurthi LIFE414 Presentation.pptx
│
├── SLIDES/                            # Course lecture materials
│   ├── Lecture6.pdf
│   ├── LIFE414_2024_Lecture1.pdf
│   ├── LIFE414_2024_Lecture2.pdf
│   ├── LIFE414_2024_Lecture3.pdf
│   ├── LIFE414_2024_Lecture4.pdf
│   ├── LIFE414_2024_Lecture5.pdf
│   ├── LIFE414_2024_Lecture6.pdf
│   ├── LIFE414_2024_Lecture7.pdf
│   ├── LIFE414_2024_Lecture8.pdf
│   ├── LIFE414_2024_Lecture9.pdf
│   └── SYS - bio notes.pdf
│
├── Part_2/                            # Additional course materials
│   └── [Lecture PDFs 1-9]
│
├── Part -2 Slides/                    # Course slides backup
│   └── [Lecture PDFs 1-9]
│
└── GWAS; Association analysis.pdf     # Additional reference material
```

## Key Features Implemented

### 1. Synthetic Biological Oscillators
- **Edelstein Relaxation Oscillator**: 
  - Complete mathematical model implementation
  - Parameter optimization using genetic algorithms
  - Noise robustness analysis and bifurcation studies
  - Real-time dynamics simulation and visualization
  
- **Otero Repressilator (3-Gene Circuit)**:
  - Full three-gene mutual repression network
  - Phase relationship analysis and coherence metrics
  - Multi-objective optimization (period + amplitude + phase coherence)
  - 3D phase portraits and comprehensive dynamics analysis

### 2. Advanced Reinforcement Learning
- **Agents**: DDPG and PPO implementations optimized for biological systems
- **Environments**: Custom biological optimization environments with:
  - Dynamic noise handling and adaptation
  - Hopf bifurcation transition management
  - Continuous parameter space exploration
  - Multi-objective reward functions
- **Training**: Advanced features including:
  - Curriculum learning with progressive difficulty
  - Robustness testing across noise levels
  - Comparative agent evaluation

### 3. Comprehensive Analysis Suite
- **Performance Metrics**: Period, amplitude, stability, phase coherence
- **Visualization Tools**: 
  - Time series and phase portraits
  - 3D dynamics visualization
  - Bifurcation diagrams
  - Interactive Plotly dashboards
  - RL training progress analysis
- **Statistical Analysis**: Robustness testing, noise sensitivity, comparative evaluation
- **Data Processing**: Advanced data cleaning, filtering, and feature extraction

## Usage and Development

### Installation and Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Run tests to verify installation
python -m pytest tests/
```

### Command-Line Usage
```bash
# Edelstein oscillator optimization
python -m edelstein_oscillator.cli simulate --k1 1.5 --plot
python -m edelstein_oscillator.cli optimize --target-period 12 --generations 100

# Otero repressilator analysis
python -m otero_repressilator.cli simulate --alpha 200 --plot
python -m otero_repressilator.cli bifurcation params.json --parameter alpha

# Reinforcement learning training
python -m reinforcement_learning.cli train --oscillator edelstein --episodes 500
python -m reinforcement_learning.cli test-robustness model.pth --noise-levels "0.0,0.1,0.2"
```

### Python API Usage
```python
# Import main components
from edelstein_oscillator import EdelsteinOscillator, EdelsteinOptimizer
from otero_repressilator import OteroRepressilator, OteroOptimizer
from reinforcement_learning import make_edelstein_env, PPO_BioAgent, RLTrainer
from analysis import ResultVisualizer, PerformanceMetrics

# Quick oscillator simulation
oscillator = EdelsteinOscillator()
result = oscillator.simulate()
oscillator.plot_trajectory()

# RL training
env = make_edelstein_env(target_period=10.0)
agent = PPO_BioAgent(state_dim=12, action_dim=6)
trainer = RLTrainer(agent, env)
trainer.train()
```

## Course Context

**Course**: Systems Biology (LIFE414)  
**Period**: August 2024 - December 2024  
**Faculty**: Dr. Jong Min Kim, Laboratory of Synthetic Biology and Molecular Computing  
**Institution**: POSTECH  

**Key Contributions**:
- Optimized Synthetic Oscillators (Edelstein Relaxation Oscillator & Otero Repressilator) using Reinforcement Learning
- Analyzed RL's ability to dynamically handle biological noise and transitions like Hopf bifurcations in biological networks
- Demonstrated superior performance of RL methods compared to traditional optimization approaches
