# BioSignal Framework

A comprehensive framework for biosignal processing, simulation, and analysis supporting EMG, ECG, and EOG signals.

## Project Overview

The BioSignal Framework is a powerful, modular toolkit designed for biosignal processing and analysis. It provides extensive capabilities for:

- Signal acquisition and simulation
- Preprocessing and noise reduction
- Feature extraction across multiple domains
- Machine learning model development
- Real-time processing and visualization

Key Features:
- Multi-signal support (EMG, ECG, EOG)
- Advanced simulation capabilities
- Comprehensive noise and artifact modeling
- Real-time processing pipeline
- Extensive feature extraction library
- Modular and extensible architecture

## Installation Instructions

### Prerequisites
- Python ≥ 3.9
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- PyWavelets (pywt)
- pandas
- IPython (for notebooks)

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/biosig.git
cd biosig
```

2. Install dependencies:
```bash
pip install numpy scipy matplotlib scikit-learn pywt pandas ipython
```

## Module Documentation

### 1. Simulation Module (`simulation.py`)

The simulation module provides comprehensive biosignal generation capabilities:

#### Base Simulator
- Configurable sampling rate and duration
- Noise addition (Gaussian, powerline)
- Artifact injection capabilities

#### Signal-Specific Simulators:

**EMG Simulator**
- Isometric contractions
- Dynamic patterns
- Fatigue modeling
- Complex movement sequences

**ECG Simulator**
- Normal sinus rhythm
- Arrhythmias (PVC, AF)
- Conduction abnormalities
- Ischemia patterns

**EOG Simulator**
- Saccadic movements
- Smooth pursuit
- Fixations
- Blink patterns

**Noise Simulator**
- Electrode artifacts
- Motion artifacts
- Environmental interference
- Powerline noise

### 2. Preprocessing Module

- Signal filtering (bandpass, notch)
- Wavelet denoising
- Normalization
- Segmentation

### 3. Feature Extraction Module

- Time domain features
- Frequency domain analysis
- Nonlinear features
- Custom feature pipelines

### 4. Models Module

- Classical ML models
- Pipeline management
- Model evaluation tools
- Real-time prediction

## Usage Examples

### 1. Basic Signal Generation
```python
from simulation import EMGSimulator

# Create EMG simulator
sim = EMGSimulator(sampling_rate=1000, duration=5.0)

# Generate basic EMG signal
signal = sim.generate(activation_level=0.7)

# Add noise and artifacts
noisy_signal = sim.add_noise(signal, noise_type='gaussian', noise_params={'std': 0.1})
signal_with_artifact = sim.add_artifact(noisy_signal, 'spike', start_time=2.0, duration=0.1)
```

### 2. Complex Signal Patterns
```python
# Generate dynamic contraction
dynamic_signal = sim.simulate_dynamic_contraction(
    pattern='ramp',
    max_intensity=0.8
)

# Generate repetitive movement
repetitive_signal = sim.simulate_repetitive_movement(
    frequency=1.0,
    intensity=0.7
)
```

### 3. Multi-Signal Simulation
```python
from simulation import ECGSimulator, EOGSimulator

# ECG with arrhythmia
ecg_sim = ECGSimulator(sampling_rate=1000)
ecg_signal = ecg_sim.simulate_arrhythmias('pvc', base_heart_rate=75)

# EOG with saccades
eog_sim = EOGSimulator(sampling_rate=1000)
eog_signal = eog_sim.simulate_saccades(amplitudes=[10, -10])
```

## Simulation Capabilities

The framework provides extensive simulation capabilities for generating realistic biosignals:

### Signal Types
1. **EMG Signals**
   - Isometric contractions
   - Dynamic patterns
   - Fatigue effects
   - Complex movement sequences

2. **ECG Signals**
   - Normal sinus rhythm
   - Various arrhythmias
   - Ischemic patterns
   - Conduction abnormalities

3. **EOG Signals**
   - Saccadic movements
   - Smooth pursuit
   - Fixations
   - Blink patterns

### Noise and Artifacts
1. **Electrode Artifacts**
   - Poor contact
   - Electrode pop
   - Impedance changes
   - DC offset variations

2. **Motion Artifacts**
   - Electrode movement
   - Cable motion
   - Subject movement
   - Baseline shifts

3. **Interference**
   - EMG crosstalk
   - ECG interference
   - Environmental noise
   - Powerline interference

## Running Tests

The framework includes comprehensive test suites for all modules:

### Running All Tests
```bash
pytest tests/
```

### Running Specific Test Modules
```bash
pytest tests/test_simulation.py
pytest tests/test_preprocessing.py
pytest tests/test_features.py
pytest tests/test_models.py
```

### Test Coverage
```bash
pytest --cov=./ tests/
```

## Demo Notebook Guide

The [`notebooks/emg_pipeline_demo.ipynb`](notebooks/emg_pipeline_demo.ipynb) notebook provides an interactive demonstration of the framework's capabilities:

### Notebook Sections:
1. **Signal Generation Explorer**
   - Interactive signal type selection
   - Parameter adjustment
   - Real-time visualization

2. **Real-time Processing Demo**
   - Filter parameter tuning
   - Noise reduction
   - Signal quality metrics

3. **Feature Visualization**
   - Time-domain features
   - Frequency-domain features
   - Nonlinear features

### Running the Demo
1. Start Jupyter Notebook:
```bash
jupyter notebook notebooks/emg_pipeline_demo.ipynb
```

2. Follow the interactive widgets to:
   - Generate different signal types
   - Adjust processing parameters
   - Visualize features
   - Experiment with noise levels

## License

MIT License © 2025 Brian Mwangi

## Citation

If you use this framework in your research, please cite:

> Mwangi, B. (2025). BioSignal Framework: A Modular Platform for EMG, ECG, and EOG Processing.
> GitHub Repository: https://github.com/Ndambia/biosignal-framework
