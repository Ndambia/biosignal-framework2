import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple

class Simulator(ABC):
    """Base class for biosignal simulation."""
    
    def __init__(self, sampling_rate: float = 1000.0, duration: float = 1.0):
        """
        Initialize simulator with basic parameters.
        
        Args:
            sampling_rate (float): Sampling frequency in Hz
            duration (float): Signal duration in seconds
        """
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.n_samples = int(self.sampling_rate * self.duration)
        self.time = np.linspace(0, self.duration, self.n_samples)
        
    def add_noise(self, signal: np.ndarray, noise_type: str = 'gaussian', 
                 noise_params: Optional[dict] = None) -> np.ndarray:
        """
        Add noise to the signal.
        
        Args:
            signal (np.ndarray): Input signal
            noise_type (str): Type of noise ('gaussian', 'powerline', etc.)
            noise_params (dict, optional): Parameters for noise generation
            
        Returns:
            np.ndarray: Signal with added noise
        """
        if noise_params is None:
            noise_params = {}
            
        if noise_type == 'gaussian':
            std = noise_params.get('std', 0.1)
            noise = np.random.normal(0, std, size=len(signal))
        elif noise_type == 'powerline':
            freq = noise_params.get('frequency', 50)  # Hz
            amplitude = noise_params.get('amplitude', 0.1)
            noise = amplitude * np.sin(2 * np.pi * freq * self.time)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
            
        return signal + noise
        
    def add_artifact(self, signal: np.ndarray, artifact_type: str,
                    start_time: float, duration: float,
                    amplitude: float = 1.0) -> np.ndarray:
        """
        Add an artifact to the signal at specified time.
        
        Args:
            signal (np.ndarray): Input signal
            artifact_type (str): Type of artifact ('spike', 'step', etc.)
            start_time (float): Start time of artifact in seconds
            duration (float): Duration of artifact in seconds
            amplitude (float): Amplitude of artifact
            
        Returns:
            np.ndarray: Signal with added artifact
        """
        start_idx = int(start_time * self.sampling_rate)
        duration_samples = int(duration * self.sampling_rate)
        end_idx = start_idx + duration_samples
        
        if artifact_type == 'spike':
            artifact = np.zeros_like(signal)
            artifact[start_idx] = amplitude
        elif artifact_type == 'step':
            artifact = np.zeros_like(signal)
            artifact[start_idx:end_idx] = amplitude
        else:
            raise ValueError(f"Unsupported artifact type: {artifact_type}")
            
        return signal + artifact
    
    @abstractmethod
    def generate(self, **kwargs) -> np.ndarray:
        """
        Generate synthetic biosignal.
        
        Returns:
            np.ndarray: Generated signal
        """
        pass

class NoiseSimulator(Simulator):
    """Class for generating various types of noise and artifacts commonly found in biosignals.
    
    This class provides comprehensive methods for simulating different types of noise,
    artifacts, and interference patterns that are commonly encountered in biosignal
    recordings. It includes:
    
    1. Various noise types (simulate_noise):
        - Gaussian white noise
        - Pink (1/f) noise
        - Brown (1/f²) noise
        - Power line interference (50/60 Hz)
        - Baseline wander
        - High-frequency noise
        
    2. Motion artifacts (simulate_motion_artifacts):
        - Electrode movement
        - Cable motion
        - Subject movement
        - Sudden baseline shifts
        
    3. Interference patterns (simulate_interference):
        - EMG crosstalk
        - ECG interference
        - Environmental electromagnetic interference
        - Device-specific artifacts
        
    4. Electrode artifacts (simulate_electrode_artifacts):
        - Poor contact
        - Electrode pop
        - Impedance changes
        - DC offset variations
    
    Example usage:
    ```python
    # Initialize simulator
    sim = NoiseSimulator(sampling_rate=1000, duration=5.0)
    
    # Generate different types of noise
    gaussian = sim.simulate_noise('gaussian', std=0.1)
    powerline = sim.simulate_noise('powerline', frequency=50, amplitude=0.2)
    baseline = sim.simulate_noise('baseline_wander', amplitude=0.5)
    
    # Generate motion artifacts
    motion = sim.simulate_motion_artifacts('electrode_movement',
                                         n_artifacts=3,
                                         amplitude=0.5)
    
    # Generate interference
    emg_crosstalk = sim.simulate_interference('emg',
                                            amplitude=0.3,
                                            n_bursts=5)
    
    # Generate electrode artifacts
    electrode_pop = sim.simulate_electrode_artifacts('electrode_pop',
                                                   amplitude=1.0,
                                                   n_events=2)
    
    # Combine multiple artifacts
    combined_noise = (gaussian + powerline + motion +
                     emg_crosstalk + electrode_pop)
    ```
    
    The simulator allows for flexible parameter configuration and can be used
    to create realistic noise and artifact patterns for testing signal processing
    algorithms, evaluating noise reduction techniques, or generating synthetic
    datasets for machine learning applications.
    """
    
    def simulate_electrode_artifacts(self, artifact_type: str = 'poor_contact', **kwargs) -> np.ndarray:
        """
        Simulate various types of electrode-related artifacts in biosignals.
        
        Args:
            artifact_type (str): Type of electrode artifact to generate:
                - 'poor_contact': Intermittent electrode contact issues
                - 'electrode_pop': Sudden electrode pop artifacts
                - 'impedance_change': Gradual changes in electrode impedance
                - 'dc_offset': DC offset variations and drift
            **kwargs: Additional parameters for artifact generation:
                - amplitude (float): Amplitude of artifacts
                - duration (float): Duration of artifacts in seconds
                - n_events (int): Number of artifact events
                - random_seed (int): Seed for random number generation
                
        Returns:
            np.ndarray: Generated electrode artifact signal
            
        Example:
            ```python
            simulator = NoiseSimulator(sampling_rate=1000, duration=5.0)
            
            # Generate poor contact artifacts
            poor_contact = simulator.simulate_electrode_artifacts(
                'poor_contact',
                amplitude=0.5,
                n_events=3
            )
            
            # Generate electrode pop artifacts
            electrode_pop = simulator.simulate_electrode_artifacts(
                'electrode_pop',
                amplitude=1.0,
                n_events=2
            )
            ```
        """
        # Set random seed if provided
        if 'random_seed' in kwargs:
            np.random.seed(kwargs['random_seed'])
            
        amplitude = kwargs.get('amplitude', 1.0)
        signal = np.zeros(self.n_samples)
        
        if artifact_type == 'poor_contact':
            # Simulate intermittent contact issues
            n_events = kwargs.get('n_events', 3)
            event_duration = kwargs.get('duration', 0.2)  # seconds
            
            for _ in range(n_events):
                start = int(np.random.uniform(0, self.duration - event_duration) * self.sampling_rate)
                duration_samples = int(event_duration * self.sampling_rate)
                
                # Generate random noise bursts and signal dropouts
                contact_quality = np.random.uniform(0, 1, duration_samples)
                noise = np.random.normal(0, amplitude, duration_samples)
                
                # Create mixture of noise and signal dropout
                artifact = np.where(contact_quality < 0.3, 0, noise)  # 30% chance of dropout
                signal[start:start + duration_samples] += artifact
                
        elif artifact_type == 'electrode_pop':
            # Simulate sudden electrode pops
            n_pops = kwargs.get('n_events', 2)
            pop_duration = kwargs.get('duration', 0.05)  # seconds
            
            for _ in range(n_pops):
                start = int(np.random.uniform(0, self.duration - pop_duration) * self.sampling_rate)
                duration_samples = int(pop_duration * self.sampling_rate)
                
                # Create sudden spike with exponential decay
                pop = amplitude * (1 if np.random.random() > 0.5 else -1)
                decay = np.exp(-np.linspace(0, 5, duration_samples))
                signal[start:start + duration_samples] += pop * decay
                
        elif artifact_type == 'impedance_change':
            # Simulate gradual changes in electrode impedance
            n_changes = kwargs.get('n_events', 2)
            change_duration = kwargs.get('duration', 0.5)  # seconds
            
            for _ in range(n_changes):
                start = int(np.random.uniform(0, self.duration - change_duration) * self.sampling_rate)
                duration_samples = int(change_duration * self.sampling_rate)
                
                # Generate smooth impedance transition
                transition = np.linspace(0, 1, duration_samples)
                # Add high-frequency noise modulated by impedance change
                noise = np.random.normal(0, 0.2 * amplitude, duration_samples)
                artifact = amplitude * transition * (1 + noise)
                signal[start:start + duration_samples] += artifact
                
        elif artifact_type == 'dc_offset':
            # Simulate DC offset variations
            base_offset = amplitude * np.random.uniform(-1, 1)
            drift_frequency = kwargs.get('drift_frequency', 0.1)  # Hz
            
            # Add slow baseline drift
            drift = amplitude * 0.5 * np.sin(2 * np.pi * drift_frequency * self.time)
            
            # Add sudden offset changes
            n_changes = kwargs.get('n_events', 3)
            for _ in range(n_changes):
                start = int(np.random.uniform(0, self.duration) * self.sampling_rate)
                offset_change = amplitude * np.random.uniform(-0.5, 0.5)
                signal[start:] += offset_change
                
            signal += base_offset + drift
            
        else:
            raise ValueError(f"Unsupported artifact type: {artifact_type}")
            
        return signal
        
    def simulate_interference(self, interference_type: str = 'emg', **kwargs) -> np.ndarray:
        """
        Simulate various types of interference commonly found in biosignals.
        
        Args:
            interference_type (str): Type of interference to generate:
                - 'emg': EMG crosstalk interference
                - 'ecg': ECG interference in other biosignals
                - 'environmental': Environmental electromagnetic interference
                - 'device': Device-specific electronic interference
            **kwargs: Additional parameters for interference generation:
                - amplitude (float): Amplitude of interference
                - frequency (float): Frequency components for periodic interference
                - duration (float): Duration of interference bursts
                - n_bursts (int): Number of interference bursts
                - random_seed (int): Seed for random number generation
                
        Returns:
            np.ndarray: Generated interference signal
            
        Example:
            ```python
            simulator = NoiseSimulator(sampling_rate=1000, duration=5.0)
            
            # Generate EMG crosstalk
            emg_interference = simulator.simulate_interference(
                'emg',
                amplitude=0.3,
                n_bursts=5
            )
            
            # Generate environmental interference
            env_interference = simulator.simulate_interference(
                'environmental',
                amplitude=0.2,
                frequency=50  # Power line frequency
            )
            ```
        """
        # Set random seed if provided
        if 'random_seed' in kwargs:
            np.random.seed(kwargs['random_seed'])
            
        amplitude = kwargs.get('amplitude', 1.0)
        signal = np.zeros(self.n_samples)
        
        if interference_type == 'emg':
            # Simulate EMG crosstalk as bursts of high-frequency activity
            n_bursts = kwargs.get('n_bursts', 5)
            burst_duration = kwargs.get('duration', 0.2)  # seconds
            
            for _ in range(n_bursts):
                start = int(np.random.uniform(0, self.duration - burst_duration) * self.sampling_rate)
                burst_samples = int(burst_duration * self.sampling_rate)
                
                # Generate burst with multiple frequency components
                t = np.linspace(0, burst_duration, burst_samples)
                burst = np.zeros(burst_samples)
                
                # Add random frequency components (20-500 Hz)
                for _ in range(10):
                    freq = np.random.uniform(20, 500)
                    phase = np.random.uniform(0, 2*np.pi)
                    burst += np.sin(2 * np.pi * freq * t + phase)
                
                # Apply amplitude modulation
                envelope = np.hanning(burst_samples)
                signal[start:start + burst_samples] += amplitude * envelope * burst
                
        elif interference_type == 'ecg':
            # Simulate ECG interference as periodic QRS-like spikes
            heart_rate = kwargs.get('heart_rate', 60)  # bpm
            interval = self.sampling_rate * 60 / heart_rate
            
            for start in range(0, self.n_samples, int(interval)):
                if start + 100 < self.n_samples:
                    # Generate QRS-like shape
                    qrs = amplitude * np.concatenate([
                        np.linspace(0, -0.2, 20),  # Q wave
                        np.linspace(-0.2, 1, 10),   # R upstroke
                        np.linspace(1, -0.3, 10),   # R downstroke
                        np.linspace(-0.3, 0, 60)    # S wave and recovery
                    ])
                    signal[start:start + 100] += qrs
                    
        elif interference_type == 'environmental':
            # Simulate environmental interference (e.g., power line, RF)
            base_freq = kwargs.get('frequency', 50)  # Power line frequency
            
            # Add fundamental frequency and harmonics
            for harmonic in range(1, 4):
                freq = base_freq * harmonic
                harmonic_amp = amplitude / harmonic
                signal += harmonic_amp * np.sin(2 * np.pi * freq * self.time)
                
            # Add random high-frequency components
            for _ in range(5):
                freq = np.random.uniform(100, 1000)
                signal += (amplitude * 0.1) * np.sin(2 * np.pi * freq * self.time)
                
        elif interference_type == 'device':
            # Simulate device-specific interference
            # Digital switching noise and periodic electronic artifacts
            
            # Add high-frequency switching noise
            switching_freq = kwargs.get('switching_freq', 1000)  # Hz
            duty_cycle = kwargs.get('duty_cycle', 0.1)
            
            # Generate square wave for switching noise
            t = np.linspace(0, self.duration, self.n_samples)
            square_wave = amplitude * 0.5 * signal.square(2 * np.pi * switching_freq * t, duty=duty_cycle)
            
            # Add random electronic spikes
            n_spikes = kwargs.get('n_spikes', 20)
            for _ in range(n_spikes):
                start = np.random.randint(0, self.n_samples - 10)
                spike_width = np.random.randint(5, 10)
                signal[start:start + spike_width] += amplitude * np.random.uniform(0.5, 1.0)
            
            # Combine effects
            signal += square_wave
            
        else:
            raise ValueError(f"Unsupported interference type: {interference_type}")
            
        return signal
        
    def simulate_motion_artifacts(self, artifact_type: str = 'electrode_movement', **kwargs) -> np.ndarray:
        """
        Simulate various types of motion artifacts commonly found in biosignals.
        
        Args:
            artifact_type (str): Type of motion artifact to generate:
                - 'electrode_movement': Sudden shifts due to electrode movement
                - 'cable_motion': Artifacts from cable movement/vibration
                - 'subject_movement': Large-scale movement artifacts
                - 'baseline_shift': Sudden baseline shifts
            **kwargs: Additional parameters for artifact generation:
                - n_artifacts (int): Number of artifacts to generate
                - amplitude (float): Amplitude of artifacts
                - duration (float): Duration of each artifact in seconds
                - random_seed (int): Seed for random number generation
                
        Returns:
            np.ndarray: Generated motion artifact signal
            
        Example:
            ```python
            simulator = NoiseSimulator(sampling_rate=1000, duration=5.0)
            
            # Generate electrode movement artifacts
            electrode_movement = simulator.simulate_motion_artifacts(
                'electrode_movement',
                n_artifacts=3,
                amplitude=0.5
            )
            
            # Generate subject movement artifacts
            subject_movement = simulator.simulate_motion_artifacts(
                'subject_movement',
                n_artifacts=2,
                amplitude=1.0,
                duration=0.5
            )
            ```
        """
        # Set random seed if provided
        if 'random_seed' in kwargs:
            np.random.seed(kwargs['random_seed'])
            
        n_artifacts = kwargs.get('n_artifacts', 3)
        amplitude = kwargs.get('amplitude', 1.0)
        duration = kwargs.get('duration', 0.2)  # seconds
        
        signal = np.zeros(self.n_samples)
        
        if artifact_type == 'electrode_movement':
            # Sudden shifts with exponential recovery
            for _ in range(n_artifacts):
                start = int(np.random.uniform(0, self.duration - duration) * self.sampling_rate)
                shift_duration = int(duration * self.sampling_rate)
                
                # Create sudden shift
                shift = amplitude * (1 if np.random.random() > 0.5 else -1)
                # Exponential recovery
                recovery = np.exp(-np.linspace(0, 5, shift_duration))
                signal[start:start + shift_duration] += shift * recovery
                
        elif artifact_type == 'cable_motion':
            # High-frequency oscillations with varying amplitude
            for _ in range(n_artifacts):
                start = int(np.random.uniform(0, self.duration - duration) * self.sampling_rate)
                artifact_duration = int(duration * self.sampling_rate)
                t = np.linspace(0, duration, artifact_duration)
                
                # Create oscillating artifact with frequency modulation
                freq = np.random.uniform(10, 30)  # Hz
                envelope = np.hanning(artifact_duration)
                artifact = amplitude * envelope * np.sin(2 * np.pi * freq * t)
                signal[start:start + artifact_duration] += artifact
                
        elif artifact_type == 'subject_movement':
            # Longer duration, complex artifacts
            for _ in range(n_artifacts):
                start = int(np.random.uniform(0, self.duration - duration) * self.sampling_rate)
                artifact_duration = int(duration * self.sampling_rate)
                t = np.linspace(0, duration, artifact_duration)
                
                # Combine multiple frequency components
                for freq in [2, 5, 8]:  # Hz
                    phase = np.random.uniform(0, 2*np.pi)
                    signal[start:start + artifact_duration] += (amplitude/3) * \
                        np.sin(2 * np.pi * freq * t + phase)
                
                # Add random baseline shift
                shift = np.random.uniform(-amplitude/2, amplitude/2)
                signal[start:start + artifact_duration] += shift
                
        elif artifact_type == 'baseline_shift':
            # Sudden baseline shifts with varying recovery
            for _ in range(n_artifacts):
                start = int(np.random.uniform(0, self.duration - duration) * self.sampling_rate)
                shift_duration = int(duration * self.sampling_rate)
                
                # Create step change
                shift = amplitude * (1 if np.random.random() > 0.5 else -1)
                signal[start:start + shift_duration] += shift
                
                # Add gradual recovery in some cases
                if np.random.random() > 0.5:
                    recovery = np.linspace(shift, 0, shift_duration)
                    signal[start:start + shift_duration] = recovery
                    
        else:
            raise ValueError(f"Unsupported artifact type: {artifact_type}")
            
        return signal
        
    def simulate_noise(self, noise_type: str = 'gaussian', **kwargs) -> np.ndarray:
        """
        Generate various types of noise commonly found in biosignals.
        
        Args:
            noise_type (str): Type of noise to generate:
                - 'gaussian': Gaussian white noise
                - 'pink': Pink (1/f) noise
                - 'brown': Brown (1/f²) noise
                - 'powerline': Power line interference (50/60 Hz)
                - 'baseline_wander': Low-frequency baseline drift
                - 'high_frequency': High-frequency noise
            **kwargs: Additional parameters for noise generation:
                - std (float): Standard deviation for Gaussian noise
                - frequency (float): Frequency for powerline interference
                - amplitude (float): Noise amplitude
                - drift_frequency (float): Frequency for baseline wander
                
        Returns:
            np.ndarray: Generated noise signal
            
        Example:
            ```python
            simulator = NoiseSimulator(sampling_rate=1000, duration=5.0)
            
            # Generate Gaussian noise
            gaussian = simulator.simulate_noise('gaussian', std=0.1)
            
            # Generate power line interference
            powerline = simulator.simulate_noise('powerline',
                                              frequency=50,
                                              amplitude=0.2)
            
            # Generate baseline wander
            baseline = simulator.simulate_noise('baseline_wander',
                                             amplitude=0.5,
                                             drift_frequency=0.2)
            ```
        """
        if noise_type == 'gaussian':
            std = kwargs.get('std', 1.0)
            return np.random.normal(0, std, size=self.n_samples)
            
        elif noise_type == 'pink':
            # Generate pink noise using 1/f power spectrum
            f = np.fft.fftfreq(self.n_samples)
            f[0] = np.inf  # Avoid division by zero
            amplitude = kwargs.get('amplitude', 1.0)
            ps = 1.0 / np.abs(f)
            ps[0] = 0
            phases = np.random.uniform(0, 2*np.pi, size=self.n_samples)
            noise = np.fft.ifft(np.sqrt(ps) * np.exp(1j*phases))
            return amplitude * noise.real
            
        elif noise_type == 'brown':
            # Generate brown noise using 1/f² power spectrum
            f = np.fft.fftfreq(self.n_samples)
            f[0] = np.inf
            amplitude = kwargs.get('amplitude', 1.0)
            ps = 1.0 / (f*f)
            ps[0] = 0
            phases = np.random.uniform(0, 2*np.pi, size=self.n_samples)
            noise = np.fft.ifft(np.sqrt(ps) * np.exp(1j*phases))
            return amplitude * noise.real
            
        elif noise_type == 'powerline':
            freq = kwargs.get('frequency', 50)  # Hz
            amplitude = kwargs.get('amplitude', 1.0)
            harmonics = kwargs.get('harmonics', 2)  # Number of harmonics to include
            noise = np.zeros(self.n_samples)
            
            # Add fundamental frequency and harmonics
            for h in range(1, harmonics + 1):
                harmonic_amp = amplitude / h  # Amplitude decreases with harmonic number
                noise += harmonic_amp * np.sin(2 * np.pi * freq * h * self.time)
            return noise
            
        elif noise_type == 'baseline_wander':
            amplitude = kwargs.get('amplitude', 1.0)
            drift_frequency = kwargs.get('drift_frequency', 0.5)  # Hz
            # Combine multiple low-frequency components
            noise = np.zeros(self.n_samples)
            for f in [drift_frequency, drift_frequency/2, drift_frequency/3]:
                noise += (amplitude/3) * np.sin(2 * np.pi * f * self.time +
                                              np.random.uniform(0, 2*np.pi))
            return noise
            
        elif noise_type == 'high_frequency':
            amplitude = kwargs.get('amplitude', 1.0)
            min_freq = kwargs.get('min_freq', 100)  # Hz
            max_freq = kwargs.get('max_freq', 500)  # Hz
            n_components = kwargs.get('n_components', 10)
            
            noise = np.zeros(self.n_samples)
            frequencies = np.random.uniform(min_freq, max_freq, n_components)
            for freq in frequencies:
                noise += (amplitude/n_components) * np.sin(2 * np.pi * freq * self.time +
                                                         np.random.uniform(0, 2*np.pi))
            return noise
            
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
    
    def generate(self, noise_type: str = 'gaussian', **kwargs) -> np.ndarray:
        """
        Generate noise signal (wrapper for simulate_noise).
        
        Args:
            noise_type (str): Type of noise to generate
            **kwargs: Additional parameters for noise generation
            
        Returns:
            np.ndarray: Generated noise signal
        """
        return self.simulate_noise(noise_type, **kwargs)

class EMGSimulator(Simulator):
    """Class for generating synthetic EMG signals.
    
    This class provides methods for simulating various types of EMG signals including:
    - Basic EMG generation with configurable parameters
    - Isometric contractions with constant force
    - Dynamic contractions with variable force patterns
    - Repetitive movements with rest periods
    - Complex movement patterns combining multiple activation types
    
    Example usage:
    ```python
    # Initialize simulator
    emg_sim = EMGSimulator(sampling_rate=1000, duration=5.0)
    
    # Generate isometric contraction with fatigue
    isometric = emg_sim.simulate_isometric_contraction(
        intensity=0.8,
        duration=3.0,
        fatigue_rate=2.0
    )
    
    # Generate dynamic contraction with ramping pattern
    dynamic = emg_sim.simulate_dynamic_contraction(
        pattern='ramp',
        max_intensity=0.9,
        burst_duration=2.0
    )
    
    # Generate repetitive movement
    repetitive = emg_sim.simulate_repetitive_movement(
        frequency=0.5,  # 2-second cycles
        duty_cycle=0.6, # 60% contraction, 40% rest
        intensity=0.7
    )
    
    # Generate complex pattern
    complex_emg = emg_sim.simulate_complex_pattern(
        movements=['isometric', 'dynamic', 'repetitive'],
        durations=[1.0, 2.0, 2.0],
        intensities=[0.8, 0.7, 0.6],
        overlap=False
    )
    
    # Add noise to any signal
    noisy_signal = emg_sim.add_noise(
        signal=isometric,
        noise_type='gaussian',
        noise_params={'std': 0.05}
    )
    ```
    """
    
    def generate_muap(self) -> np.ndarray:
        """Generate a single Motor Unit Action Potential."""
        t = np.linspace(-0.002, 0.002, int(0.004 * self.sampling_rate))
        muap = -t * np.exp(-2000 * t**2)
        return muap
    
    def generate(self, activation_level: float = 0.5, 
                fatigue: bool = False, **kwargs) -> np.ndarray:
        """
        Generate synthetic EMG signal.
        
        Args:
            activation_level (float): Muscle activation level (0 to 1)
            fatigue (bool): Whether to simulate fatigue effects
            **kwargs: Additional parameters
            
        Returns:
            np.ndarray: Generated EMG signal
        """
        # Base signal
        signal = np.zeros(self.n_samples)
        
        # Generate MUAPs
        muap = self.generate_muap()
        muap_rate = int(50 + 450 * activation_level)  # Hz
        n_muaps = int(muap_rate * self.duration)
        
        # Randomly distribute MUAPs
        for _ in range(n_muaps):
            start_idx = np.random.randint(0, self.n_samples - len(muap))
            amplitude = np.random.uniform(0.5, 1.0)
            signal[start_idx:start_idx + len(muap)] += amplitude * muap
            
        if fatigue:
            # Simulate fatigue by gradually decreasing amplitude and frequency content
            fatigue_envelope = np.exp(-2 * self.time / self.duration)
            signal *= fatigue_envelope
            
        return signal
        
    def simulate_dynamic_contraction(self, pattern: Union[str, np.ndarray],
                                   max_intensity: float = 0.8,
                                   burst_duration: Optional[float] = None) -> np.ndarray:
        """
        Simulate dynamic muscle contractions with variable force levels.
        
        Args:
            pattern (Union[str, np.ndarray]): Either a predefined pattern name ('ramp', 'sine')
                                            or custom intensity envelope array
            max_intensity (float): Maximum contraction intensity (0 to 1)
            burst_duration (float, optional): Duration of each burst in seconds
                                           Only used for burst patterns
                                    
        Returns:
            np.ndarray: Simulated EMG signal with dynamic force pattern
        """
        if isinstance(pattern, str):
            time = np.linspace(0, self.duration, self.n_samples)
            if pattern == 'ramp':
                intensity_envelope = (time / self.duration) * max_intensity
            elif pattern == 'sine':
                freq = 1.0  # 1 Hz oscillation
                intensity_envelope = (max_intensity/2) * (1 + np.sin(2 * np.pi * freq * time))
            else:
                raise ValueError(f"Unsupported pattern type: {pattern}")
        else:
            # Resample provided pattern to match signal length
            intensity_envelope = np.interp(
                np.linspace(0, 1, self.n_samples),
                np.linspace(0, 1, len(pattern)),
                pattern
            )
            
        signal = np.zeros(self.n_samples)
        muap = self.generate_muap()
        
        # Generate variable-rate MUAPs based on intensity envelope
        for i in range(self.n_samples):
            if i >= len(signal) - len(muap):
                break
                
            current_intensity = intensity_envelope[i]
            # Firing rate varies with intensity
            inst_rate = int(50 + 450 * current_intensity)
            
            # Probabilistic MUAP generation
            if np.random.random() < inst_rate / self.sampling_rate:
                amplitude = 0.7 + 0.3 * current_intensity
                amplitude *= np.random.uniform(0.9, 1.1)  # Add small variation
                signal[i:i + len(muap)] += amplitude * muap
                
        return signal
        
    def simulate_repetitive_movement(self, frequency: float = 1.0,
                                   duty_cycle: float = 0.5,
                                   intensity: float = 0.7,
                                   rest_intensity: float = 0.1) -> np.ndarray:
        """
        Simulate cyclic muscle contractions with rest periods.
        
        Args:
            frequency (float): Movement frequency in Hz
            duty_cycle (float): Fraction of cycle spent in contraction (0 to 1)
            intensity (float): Contraction intensity level (0 to 1)
            rest_intensity (float): Background activity during rest (0 to 1)
                                    
        Returns:
            np.ndarray: Simulated EMG signal for repetitive movement
        """
        period = 1.0 / frequency
        n_cycles = int(self.duration / period)
        signal = np.zeros(self.n_samples)
        
        for cycle in range(n_cycles):
            cycle_start = int(cycle * period * self.sampling_rate)
            contraction_samples = int(period * duty_cycle * self.sampling_rate)
            
            # Active contraction period
            if cycle_start + contraction_samples <= self.n_samples:
                contraction = self.simulate_isometric_contraction(
                    intensity=intensity,
                    duration=period * duty_cycle
                )
                signal[cycle_start:cycle_start + len(contraction)] += contraction
            
            # Rest period
            rest_start = cycle_start + contraction_samples
            rest_end = cycle_start + int(period * self.sampling_rate)
            if rest_start < self.n_samples:
                rest_end = min(rest_end, self.n_samples)
                rest_duration = (rest_end - rest_start) / self.sampling_rate
                
                if rest_duration > 0:
                    rest_signal = self.simulate_isometric_contraction(
                        intensity=rest_intensity,
                        duration=rest_duration
                    )
                    signal[rest_start:rest_start + len(rest_signal)] += rest_signal
                    
        return signal
        
    def simulate_complex_pattern(self, movements: list,
                               durations: list,
                               intensities: list,
                               overlap: bool = False) -> np.ndarray:
        """
        Simulate complex movement patterns with multiple muscle activations.
        
        Args:
            movements (list): List of movement types ('isometric', 'dynamic', 'repetitive')
            durations (list): List of movement durations in seconds
            intensities (list): List of movement intensities (0 to 1)
            overlap (bool): Whether to allow movement pattern overlap
                                    
        Returns:
            np.ndarray: Simulated EMG signal with complex movement pattern
        """
        if not (len(movements) == len(durations) == len(intensities)):
            raise ValueError("movements, durations, and intensities must have same length")
            
        total_duration = sum(durations) if not overlap else max(durations)
        if total_duration > self.duration:
            raise ValueError("Total movement duration exceeds simulator duration")
            
        signal = np.zeros(self.n_samples)
        current_time = 0
        
        for mov_type, dur, intens in zip(movements, durations, intensities):
            start_idx = int(current_time * self.sampling_rate)
            
            if mov_type == 'isometric':
                segment = self.simulate_isometric_contraction(
                    intensity=intens,
                    duration=dur
                )
            elif mov_type == 'dynamic':
                segment = self.simulate_dynamic_contraction(
                    pattern='ramp',
                    max_intensity=intens,
                    burst_duration=dur
                )
            elif mov_type == 'repetitive':
                segment = self.simulate_repetitive_movement(
                    frequency=1.0,
                    intensity=intens,
                    duty_cycle=0.5,
                    rest_intensity=0.1
                )
            else:
                raise ValueError(f"Unsupported movement type: {mov_type}")
                
            end_idx = start_idx + len(segment)
            if end_idx <= self.n_samples:
                if overlap:
                    signal[start_idx:end_idx] += segment
                else:
                    signal[start_idx:end_idx] = segment
                    current_time += dur
            
        return signal
                                     duration: Optional[float] = None,
                                     fatigue_rate: Optional[float] = None) -> np.ndarray:
        """
        Simulate an isometric muscle contraction with constant force level.
        
        Args:
            intensity (float): Contraction intensity level between 0 and 1
            duration (float, optional): Duration of contraction in seconds.
                                     If None, uses simulator duration
            fatigue_rate (float, optional): Rate of amplitude decay due to fatigue.
                                          If None, no fatigue effect is applied
                                    
        Returns:
            np.ndarray: Simulated EMG signal for isometric contraction
        """
        if duration is None:
            duration = self.duration
            
        n_samples = int(duration * self.sampling_rate)
        signal = np.zeros(n_samples)
        
        # Generate base MUAP template
        muap = self.generate_muap()
        
        # Calculate firing rate based on intensity
        base_rate = 50  # Minimum firing rate
        max_rate_increase = 450  # Maximum additional rate
        muap_rate = int(base_rate + max_rate_increase * intensity)
        n_muaps = int(muap_rate * duration)
        
        # Generate MUAPs with consistent amplitude for isometric contraction
        base_amplitude = 0.7 + 0.3 * intensity  # Scale amplitude with intensity
        for _ in range(n_muaps):
            start_idx = np.random.randint(0, n_samples - len(muap))
            # Add small amplitude variation (±10%) around base amplitude
            amplitude = base_amplitude * np.random.uniform(0.9, 1.1)
            signal[start_idx:start_idx + len(muap)] += amplitude * muap
            
        # Apply fatigue effect if specified
        if fatigue_rate is not None:
            time = np.linspace(0, duration, n_samples)
            fatigue_envelope = np.exp(-fatigue_rate * time / duration)
            signal *= fatigue_envelope
            
        return signal

class ECGSimulator(Simulator):
    """Class for generating synthetic ECG signals with various cardiac conditions.
    
    This class provides comprehensive methods for simulating various types of ECG signals including:
    - Normal sinus rhythm with configurable wave morphology and heart rate variability
    - Various cardiac arrhythmias (PVCs, AF, bradycardia, tachycardia, heart blocks)
    - Ischemic changes (ST elevation/depression, T wave inversion, Q waves)
    - Conduction abnormalities (bundle branch blocks, WPW syndrome, fascicular blocks)
    
    Example usage:
    ```python
    # Initialize simulator
    ecg_sim = ECGSimulator(sampling_rate=1000, duration=10.0)
    
    # Generate normal sinus rhythm with HRV
    normal_sinus = ecg_sim.simulate_normal_sinus(
        heart_rate=75.0,
        hrv_std=0.05,
        p_wave_params={'amplitude': 0.25, 'duration': 0.08},
        qrs_params={'r_amp': 1.2, 'duration': 0.08},
        t_wave_params={'amplitude': 0.35, 'duration': 0.12}
    )
    
    # Generate PVC arrhythmia
    pvc_signal = ecg_sim.simulate_arrhythmias(
        arrhythmia_type='pvc',
        base_heart_rate=80.0,
        pvc_frequency=0.2
    )
    
    # Generate STEMI
    stemi = ecg_sim.simulate_ischemia(
        ischemia_type='st_elevation',
        severity=0.8,
        heart_rate=90.0
    )
    
    # Generate LBBB
    lbbb = ecg_sim.simulate_conduction_abnormalities(
        abnormality_type='lbbb',
        heart_rate=75.0,
        severity=0.7
    )
    
    # Add noise or artifacts to any signal
    noisy_signal = ecg_sim.add_noise(
        signal=normal_sinus,
        noise_type='gaussian',
        noise_params={'std': 0.05}
    )
    ```
    """
    
    def simulate_conduction_abnormalities(self,
                                        abnormality_type: str,
                                        heart_rate: float = 75.0,
                                        severity: float = 0.5) -> np.ndarray:
        """
        Generate ECG signal with various conduction abnormalities.
        
        Args:
            abnormality_type (str): Type of conduction abnormality:
                - 'lbbb': Left Bundle Branch Block
                - 'rbbb': Right Bundle Branch Block
                - 'wpw': Wolff-Parkinson-White syndrome
                - 'lafb': Left Anterior Fascicular Block
            heart_rate (float): Heart rate in BPM
            severity (float): Severity of conduction abnormality (0-1)
        
        Returns:
            np.ndarray: Simulated ECG signal with conduction abnormalities
        """
        signal = np.zeros(self.n_samples)
        base_interval = 60.0 / heart_rate
        n_beats = int(self.duration / base_interval)
        
        if abnormality_type in ['lbbb', 'rbbb']:
            # Bundle branch blocks: widened QRS with specific morphology
            for i in range(n_beats):
                beat_start = int(i * base_interval * self.sampling_rate)
                
                # Add P wave (normal conduction to atria)
                p_wave, p_offset = self.generate_waveform('p')
                p_start = beat_start + int(p_offset * self.sampling_rate)
                if p_start >= 0 and p_start + len(p_wave) < self.n_samples:
                    signal[p_start:p_start + len(p_wave)] += p_wave
                
                # Generate modified QRS for bundle branch block
                qrs_duration = 0.12 + severity * 0.08  # Widened QRS (120-200ms)
                t = np.linspace(-qrs_duration/2, qrs_duration/2,
                              int(qrs_duration * self.sampling_rate))
                
                if abnormality_type == 'lbbb':
                    # LBBB: broad, notched R wave in V6
                    qrs = (0.8 * np.exp(-50 * (t + qrs_duration/4)**2) +
                          np.exp(-50 * t**2) +
                          0.8 * np.exp(-50 * (t - qrs_duration/4)**2))
                else:  # RBBB
                    # RBBB: RSR' pattern in V1
                    qrs = (-0.5 * np.exp(-50 * (t + qrs_duration/4)**2) +
                          np.exp(-50 * t**2) +
                          0.7 * np.exp(-50 * (t - qrs_duration/4)**2))
                
                qrs_start = beat_start + int(0.04 * self.sampling_rate)  # PR interval
                if qrs_start >= 0 and qrs_start + len(qrs) < self.n_samples:
                    signal[qrs_start:qrs_start + len(qrs)] += qrs
                
                # Add T wave
                t_wave, t_offset = self.generate_waveform('t')
                t_start = qrs_start + len(qrs) + int(0.05 * self.sampling_rate)
                if t_start >= 0 and t_start + len(t_wave) < self.n_samples:
                    signal[t_start:t_start + len(t_wave)] += t_wave
                    
        elif abnormality_type == 'wpw':
            # Wolff-Parkinson-White: short PR and delta wave
            for i in range(n_beats):
                beat_start = int(i * base_interval * self.sampling_rate)
                
                # Add P wave
                p_wave, _ = self.generate_waveform('p')
                p_start = beat_start
                if p_start >= 0 and p_start + len(p_wave) < self.n_samples:
                    signal[p_start:p_start + len(p_wave)] += p_wave
                
                # Generate delta wave
                delta_duration = 0.04 * severity  # 40ms max
                delta_samples = int(delta_duration * self.sampling_rate)
                delta_wave = np.linspace(0, severity * 0.3, delta_samples)  # Slurred upstroke
                
                delta_start = beat_start + int(0.08 * self.sampling_rate)  # Short PR
                if delta_start >= 0 and delta_start + len(delta_wave) < self.n_samples:
                    signal[delta_start:delta_start + len(delta_wave)] += delta_wave
                
                # Modified QRS complex
                qrs_wave, _ = self.generate_waveform('qrs')
                qrs_start = delta_start + len(delta_wave)
                if qrs_start >= 0 and qrs_start + len(qrs_wave) < self.n_samples:
                    signal[qrs_start:qrs_start + len(qrs_wave)] += qrs_wave
                
                # Add T wave
                t_wave, t_offset = self.generate_waveform('t')
                t_start = qrs_start + len(qrs_wave) + int(0.05 * self.sampling_rate)
                if t_start >= 0 and t_start + len(t_wave) < self.n_samples:
                    signal[t_start:t_start + len(t_wave)] += t_wave
                    
        elif abnormality_type == 'lafb':
            # Left Anterior Fascicular Block
            for i in range(n_beats):
                beat_start = int(i * base_interval * self.sampling_rate)
                
                # Normal P wave
                p_wave, p_offset = self.generate_waveform('p')
                p_start = beat_start + int(p_offset * self.sampling_rate)
                if p_start >= 0 and p_start + len(p_wave) < self.n_samples:
                    signal[p_start:p_start + len(p_wave)] += p_wave
                
                # Modified QRS: Left axis deviation with small q, tall R, small s
                qrs_duration = 0.08 + severity * 0.04  # Slightly widened
                t = np.linspace(-qrs_duration/2, qrs_duration/2,
                              int(qrs_duration * self.sampling_rate))
                
                qrs = (-0.2 * np.exp(-50 * (t + qrs_duration/4)**2) +  # small q
                       1.5 * np.exp(-50 * t**2) +                      # tall R
                       -0.3 * np.exp(-50 * (t - qrs_duration/4)**2))   # small s
                
                qrs_start = beat_start + int(0.16 * self.sampling_rate)  # Normal PR
                if qrs_start >= 0 and qrs_start + len(qrs) < self.n_samples:
                    signal[qrs_start:qrs_start + len(qrs)] += qrs
                
                # Normal T wave
                t_wave, t_offset = self.generate_waveform('t')
                t_start = qrs_start + len(qrs) + int(0.05 * self.sampling_rate)
                if t_start >= 0 and t_start + len(t_wave) < self.n_samples:
                    signal[t_start:t_start + len(t_wave)] += t_wave
                    
        return signal
    
    def simulate_ischemia(self,
                         ischemia_type: str,
                         severity: float = 0.5,
                         heart_rate: float = 75.0,
                         affected_leads: Optional[list] = None) -> np.ndarray:
        """
        Generate ECG signal with ischemic changes.
        
        Args:
            ischemia_type (str): Type of ischemic changes:
                - 'st_elevation': ST segment elevation (STEMI)
                - 'st_depression': ST segment depression
                - 't_wave_inversion': Inverted T waves
                - 'q_wave': Pathological Q waves
            severity (float): Severity of ischemic changes (0-1)
            heart_rate (float): Heart rate in BPM
            affected_leads (list, optional): List of affected leads (for multi-lead simulation)
        
        Returns:
            np.ndarray: Simulated ECG signal with ischemic changes
        """
        # Generate base normal sinus rhythm
        signal = self.simulate_normal_sinus(heart_rate=heart_rate)
        
        # Calculate beat parameters
        base_interval = 60.0 / heart_rate
        n_beats = int(self.duration / base_interval)
        
        if ischemia_type == 'st_elevation':
            # Modify each beat to show ST elevation
            for i in range(n_beats):
                beat_start = int(i * base_interval * self.sampling_rate)
                
                # Add elevated ST segment
                st_duration = 0.1  # 100ms
                st_samples = int(st_duration * self.sampling_rate)
                st_start = beat_start + int(0.1 * self.sampling_rate)  # Start after QRS
                
                if st_start >= 0 and st_start + st_samples < self.n_samples:
                    # Create elevated ST segment
                    elevation = severity * 0.3  # Max 0.3mV elevation
                    signal[st_start:st_start + st_samples] += elevation
                    
        elif ischemia_type == 'st_depression':
            # Similar to elevation but negative displacement
            for i in range(n_beats):
                beat_start = int(i * base_interval * self.sampling_rate)
                st_duration = 0.1
                st_samples = int(st_duration * self.sampling_rate)
                st_start = beat_start + int(0.1 * self.sampling_rate)
                
                if st_start >= 0 and st_start + st_samples < self.n_samples:
                    depression = -severity * 0.2  # Max 0.2mV depression
                    signal[st_start:st_start + st_samples] += depression
                    
        elif ischemia_type == 't_wave_inversion':
            # Invert and modify T waves
            for i in range(n_beats):
                beat_start = int(i * base_interval * self.sampling_rate)
                t_wave, t_offset = self.generate_waveform('t')
                t_start = beat_start + int(0.2 * self.sampling_rate)
                
                if t_start >= 0 and t_start + len(t_wave) < self.n_samples:
                    # Remove original T wave and add inverted one
                    inverted_t = -severity * t_wave
                    signal[t_start:t_start + len(t_wave)] = inverted_t
                    
        elif ischemia_type == 'q_wave':
            # Add pathological Q waves
            for i in range(n_beats):
                beat_start = int(i * base_interval * self.sampling_rate)
                
                # Generate deep Q wave
                q_duration = 0.04  # 40ms
                q_samples = int(q_duration * self.sampling_rate)
                q_wave = -severity * 0.4 * np.ones(q_samples)  # Deep Q wave
                
                # Add Q wave before QRS complex
                q_start = beat_start - q_samples
                if q_start >= 0 and q_start + q_samples < self.n_samples:
                    signal[q_start:q_start + q_samples] = q_wave
                    
        return signal
    
    def simulate_arrhythmias(self,
                            arrhythmia_type: str,
                            base_heart_rate: float = 75.0,
                            pvc_frequency: float = 0.2,
                            af_rate: Tuple[float, float] = (350, 600),
                            heart_block_degree: int = 1) -> np.ndarray:
        """
        Generate various types of cardiac arrhythmias.
        
        Args:
            arrhythmia_type (str): Type of arrhythmia to simulate:
                - 'pvc': Premature Ventricular Contractions
                - 'af': Atrial Fibrillation
                - 'brady': Bradycardia
                - 'tachy': Tachycardia
                - 'heart_block': Various degrees of AV block
            base_heart_rate (float): Base heart rate in BPM
            pvc_frequency (float): Frequency of PVCs (0-1) for PVC simulation
            af_rate (Tuple[float, float]): Range of atrial rates for AF (min, max)
            heart_block_degree (int): Degree of heart block (1, 2, or 3)
        
        Returns:
            np.ndarray: Simulated ECG signal with specified arrhythmia
        """
        signal = np.zeros(self.n_samples)
        
        if arrhythmia_type == 'pvc':
            # Generate normal rhythm with PVCs
            base_interval = 60.0 / base_heart_rate
            n_beats = int(self.duration / base_interval)
            
            for i in range(n_beats):
                beat_start = int(i * base_interval * self.sampling_rate)
                
                if np.random.random() < pvc_frequency:
                    # Generate PVC
                    pvc_wave = 2.5 * self.generate_waveform('qrs')[0]  # Larger amplitude
                    if beat_start >= 0 and beat_start + len(pvc_wave) < self.n_samples:
                        signal[beat_start:beat_start + len(pvc_wave)] += pvc_wave
                else:
                    # Normal beat
                    self._add_normal_beat(signal, beat_start)
                    
        elif arrhythmia_type == 'af':
            # Simulate atrial fibrillation
            # Irregular ventricular rhythm with no P waves
            current_time = 0
            while current_time < self.duration:
                # Random RR interval for AF
                rr_interval = np.random.uniform(60/af_rate[1], 60/af_rate[0])
                beat_start = int(current_time * self.sampling_rate)
                
                # Add QRS and T wave only (no P wave in AF)
                qrs_wave, _ = self.generate_waveform('qrs')
                t_wave, t_offset = self.generate_waveform('t')
                
                if beat_start >= 0 and beat_start + len(qrs_wave) < self.n_samples:
                    signal[beat_start:beat_start + len(qrs_wave)] += qrs_wave
                    
                t_start = beat_start + int(t_offset * self.sampling_rate)
                if t_start >= 0 and t_start + len(t_wave) < self.n_samples:
                    signal[t_start:t_start + len(t_wave)] += t_wave
                    
                current_time += rr_interval
                
        elif arrhythmia_type in ['brady', 'tachy']:
            # Simulate bradycardia (<60 BPM) or tachycardia (>100 BPM)
            hr = 45 if arrhythmia_type == 'brady' else 120
            signal = self.simulate_normal_sinus(heart_rate=hr)
            
        elif arrhythmia_type == 'heart_block':
            base_interval = 60.0 / base_heart_rate
            n_beats = int(self.duration / base_interval)
            
            for i in range(n_beats):
                beat_start = int(i * base_interval * self.sampling_rate)
                p_wave, p_offset = self.generate_waveform('p')
                
                # Always conduct P waves
                p_start = beat_start + int(p_offset * self.sampling_rate)
                if p_start >= 0 and p_start + len(p_wave) < self.n_samples:
                    signal[p_start:p_start + len(p_wave)] += p_wave
                
                # Conduct QRS based on block degree
                if heart_block_degree == 1:
                    # First-degree: prolonged PR interval
                    qrs_delay = 0.3  # 300ms PR interval
                elif heart_block_degree == 2:
                    # Second-degree: occasional dropped beats
                    qrs_delay = 0.2 if i % 2 == 0 else None
                else:  # Third-degree
                    # Complete dissociation
                    if i % 3 == 0:  # Slower ventricular escape rhythm
                        qrs_delay = np.random.uniform(0.2, 0.4)
                    else:
                        qrs_delay = None
                        
                if qrs_delay is not None:
                    qrs_start = beat_start + int(qrs_delay * self.sampling_rate)
                    self._add_normal_beat(signal, qrs_start, include_p=False)
                    
        return signal
        
    def _add_normal_beat(self, signal: np.ndarray, beat_start: int, include_p: bool = True):
        """Helper method to add a normal beat to the signal."""
        if include_p:
            p_wave, p_offset = self.generate_waveform('p')
            p_start = beat_start + int(p_offset * self.sampling_rate)
            if p_start >= 0 and p_start + len(p_wave) < self.n_samples:
                signal[p_start:p_start + len(p_wave)] += p_wave
                
        qrs_wave, qrs_offset = self.generate_waveform('qrs')
        t_wave, t_offset = self.generate_waveform('t')
        
        qrs_start = beat_start + int(qrs_offset * self.sampling_rate)
        if qrs_start >= 0 and qrs_start + len(qrs_wave) < self.n_samples:
            signal[qrs_start:qrs_start + len(qrs_wave)] += qrs_wave
            
        t_start = beat_start + int(t_offset * self.sampling_rate)
        if t_start >= 0 and t_start + len(t_wave) < self.n_samples:
            signal[t_start:t_start + len(t_wave)] += t_wave
    
    def simulate_normal_sinus(self,
                            heart_rate: float = 75.0,
                            p_wave_params: Optional[dict] = None,
                            qrs_params: Optional[dict] = None,
                            t_wave_params: Optional[dict] = None,
                            hrv_std: float = 0.0) -> np.ndarray:
        """
        Generate normal sinus rhythm with configurable wave morphology and heart rate variability.
        
        Args:
            heart_rate (float): Base heart rate in BPM (60-100 for normal sinus)
            p_wave_params (dict, optional): Parameters for P wave morphology
                - amplitude (float): P wave amplitude (default: 0.2)
                - duration (float): P wave duration in seconds (default: 0.1)
            qrs_params (dict, optional): Parameters for QRS complex
                - q_amp (float): Q wave amplitude (default: -0.5)
                - r_amp (float): R wave amplitude (default: 1.0)
                - s_amp (float): S wave amplitude (default: -0.2)
                - duration (float): QRS duration in seconds (default: 0.1)
            t_wave_params (dict, optional): Parameters for T wave morphology
                - amplitude (float): T wave amplitude (default: 0.3)
                - duration (float): T wave duration in seconds (default: 0.14)
            hrv_std (float): Standard deviation of heart rate variability in seconds
                          (0.0 for no variability)
        
        Returns:
            np.ndarray: Simulated normal sinus rhythm ECG signal
        """
        # Initialize default parameters
        p_params = {
            'amplitude': 0.2,
            'duration': 0.1
        }
        if p_wave_params:
            p_params.update(p_wave_params)
            
        qrs_p = {
            'q_amp': -0.5,
            'r_amp': 1.0,
            's_amp': -0.2,
            'duration': 0.1
        }
        if qrs_params:
            qrs_p.update(qrs_params)
            
        t_params = {
            'amplitude': 0.3,
            'duration': 0.14
        }
        if t_wave_params:
            t_params.update(t_wave_params)
            
        signal = np.zeros(self.n_samples)
        
        # Generate wave components with custom parameters
        p_wave = p_params['amplitude'] * np.exp(-100 * (np.linspace(-p_params['duration']/2,
                                                                   p_params['duration']/2,
                                                                   int(p_params['duration'] * self.sampling_rate)))**2)
        
        t = np.linspace(-qrs_p['duration']/2, qrs_p['duration']/2, int(qrs_p['duration'] * self.sampling_rate))
        qrs_wave = (qrs_p['q_amp'] * np.exp(-50 * (t + qrs_p['duration']/4)**2) +
                   qrs_p['r_amp'] * np.exp(-50 * t**2) +
                   qrs_p['s_amp'] * np.exp(-50 * (t - qrs_p['duration']/4)**2))
        
        t_wave = t_params['amplitude'] * np.exp(-100 * (np.linspace(-t_params['duration']/2,
                                                                   t_params['duration']/2,
                                                                   int(t_params['duration'] * self.sampling_rate)))**2)
        
        # Calculate intervals
        base_interval = 60.0 / heart_rate  # seconds
        n_beats = int(self.duration / base_interval)
        
        # Generate beats with HRV
        for i in range(n_beats):
            # Add heart rate variability
            if hrv_std > 0:
                beat_interval = base_interval + np.random.normal(0, hrv_std)
            else:
                beat_interval = base_interval
                
            beat_start = int(i * beat_interval * self.sampling_rate)
            
            # Add P wave
            p_start = beat_start - int(0.2 * self.sampling_rate)  # P wave starts 200ms before QRS
            if p_start >= 0 and p_start + len(p_wave) < self.n_samples:
                signal[p_start:p_start + len(p_wave)] += p_wave
                
            # Add QRS complex
            if beat_start >= 0 and beat_start + len(qrs_wave) < self.n_samples:
                signal[beat_start:beat_start + len(qrs_wave)] += qrs_wave
                
            # Add T wave
            t_start = beat_start + int(0.2 * self.sampling_rate)  # T wave starts 200ms after QRS
            if t_start >= 0 and t_start + len(t_wave) < self.n_samples:
                signal[t_start:t_start + len(t_wave)] += t_wave
                
        return signal
    
    def generate_waveform(self, wave_type: str) -> Tuple[np.ndarray, float]:
        """Generate individual ECG waveform components."""
        if wave_type == 'p':
            t = np.linspace(-0.05, 0.05, int(0.1 * self.sampling_rate))
            wave = 0.2 * np.exp(-100 * t**2)
            offset = -0.2
        elif wave_type == 'qrs':
            t = np.linspace(-0.05, 0.05, int(0.1 * self.sampling_rate))
            wave = -0.5 * np.exp(-50 * (t + 0.025)**2) + np.exp(-50 * t**2) - 0.2 * np.exp(-50 * (t - 0.025)**2)
            offset = 0
        elif wave_type == 't':
            t = np.linspace(-0.07, 0.07, int(0.14 * self.sampling_rate))
            wave = 0.3 * np.exp(-100 * t**2)
            offset = 0.2
        return wave, offset
    
    def generate(self, condition: Optional[str] = None, **kwargs) -> np.ndarray:
        """
        Generate synthetic ECG signal with specified cardiac condition.
        
        Args:
            condition (str, optional): Cardiac condition to simulate. Can be:
                - None: Normal sinus rhythm (uses simulate_normal_sinus)
                - 'pvc', 'af', 'brady', 'tachy', 'heart_block': Arrhythmias
                - 'st_elevation', 'st_depression', 't_wave_inversion', 'q_wave': Ischemia
                - 'lbbb', 'rbbb', 'wpw', 'lafb': Conduction abnormalities
            **kwargs: Additional parameters passed to specific simulation methods:
                - heart_rate (float): Base heart rate in BPM
                - severity (float): Severity of abnormality (0-1)
                - pvc_frequency (float): Frequency of PVCs (0-1)
                - af_rate (Tuple[float, float]): Range of atrial rates for AF
                - heart_block_degree (int): Degree of heart block (1, 2, or 3)
                - hrv_std (float): Heart rate variability standard deviation
                - p_wave_params (dict): P wave morphology parameters
                - qrs_params (dict): QRS complex parameters
                - t_wave_params (dict): T wave morphology parameters
            
        Returns:
            np.ndarray: Generated ECG signal with specified condition
            
        Example:
            ```python
            # Generate normal sinus rhythm
            normal = ecg.generate(heart_rate=75.0, hrv_std=0.05)
            
            # Generate PVC arrhythmia
            pvc = ecg.generate('pvc', heart_rate=80.0, pvc_frequency=0.2)
            
            # Generate STEMI
            stemi = ecg.generate('st_elevation', severity=0.8, heart_rate=90.0)
            
            # Generate LBBB
            lbbb = ecg.generate('lbbb', severity=0.7, heart_rate=75.0)
            ```
        """
        if condition is None:
            return self.simulate_normal_sinus(**kwargs)
            
        # Arrhythmias
        elif condition in ['pvc', 'af', 'brady', 'tachy', 'heart_block']:
            return self.simulate_arrhythmias(arrhythmia_type=condition, **kwargs)
            
        # Ischemic changes
        elif condition in ['st_elevation', 'st_depression', 't_wave_inversion', 'q_wave']:
            return self.simulate_ischemia(ischemia_type=condition, **kwargs)
            
        # Conduction abnormalities
        elif condition in ['lbbb', 'rbbb', 'wpw', 'lafb']:
            return self.simulate_conduction_abnormalities(abnormality_type=condition, **kwargs)
            
        else:
            raise ValueError(f"Unsupported cardiac condition: {condition}")

class EOGSimulator(Simulator):
    """Class for generating synthetic EOG signals with various eye movement patterns.
    
    This class provides methods for simulating different types of eye movements including:
    - Saccades (rapid eye movements)
    - Smooth pursuit movements
    - Fixations with microsaccades and drift
    - Natural blink patterns
    
    Example usage:
    ```python
    # Initialize simulator
    eog_sim = EOGSimulator(sampling_rate=1000, duration=5.0)
    
    # Generate horizontal saccades
    saccades = eog_sim.simulate_saccades(
        amplitudes=[10, -15, 20],  # degrees
        directions=['horizontal'] * 3,
        durations=[0.05, 0.06, 0.07]  # seconds
    )
    
    # Generate smooth pursuit
    pursuit = eog_sim.simulate_smooth_pursuit(
        pattern='circular',
        frequency=0.5,  # Hz
        amplitude=10  # degrees
    )
    
    # Generate fixation with microsaccades
    fixation = eog_sim.simulate_fixations(
        duration=2.0,
        microsaccade_rate=2.0,  # Hz
        drift_amplitude=0.5  # degrees
    )
    
    # Generate natural blinks
    blinks = eog_sim.simulate_blinks(
        n_blinks=3,
        blink_duration=0.2,  # seconds
        amplitude_range=(0.8, 1.2)
    )
    ```
    """
    
    def simulate_saccades(self, amplitudes: Union[float, List[float]],
                         directions: Union[str, List[str]] = 'horizontal',
                         durations: Optional[Union[float, List[float]]] = None,
                         peak_velocities: Optional[Union[float, List[float]]] = None) -> np.ndarray:
        """Generate saccadic eye movements with configurable parameters.
        
        Args:
            amplitudes (Union[float, List[float]]): Saccade amplitude(s) in degrees.
                Single value or list for multiple saccades.
            directions (Union[str, List[str]]): Movement direction(s) ('horizontal' or 'vertical').
                Single value or list matching amplitudes length.
            durations (Optional[Union[float, List[float]]]): Saccade duration(s) in seconds.
                If None, calculated using main sequence relationship.
            peak_velocities (Optional[Union[float, List[float]]]): Peak velocity override(s) in deg/s.
                If None, calculated using main sequence relationship.
                
        Returns:
            np.ndarray: Generated EOG signal containing specified saccades
            
        Notes:
            - Uses the main sequence relationship to determine realistic durations and velocities
            - Implements a physiologically-based velocity profile (asymmetric)
            - Supports both horizontal and vertical movements
            - Automatically spaces saccades to avoid overlap
        """
        # Convert single values to lists
        if isinstance(amplitudes, (int, float)):
            amplitudes = [amplitudes]
        if isinstance(directions, str):
            directions = [directions] * len(amplitudes)
        if durations is not None and isinstance(durations, (int, float)):
            durations = [durations] * len(amplitudes)
        if peak_velocities is not None and isinstance(peak_velocities, (int, float)):
            peak_velocities = [peak_velocities] * len(amplitudes)
            
        if len(directions) != len(amplitudes):
            raise ValueError("Number of directions must match number of amplitudes")
            
        signal = np.zeros(self.n_samples)
        current_time = 0.0
        
        for i, (amplitude, direction) in enumerate(zip(amplitudes, directions)):
            # Calculate duration using main sequence if not provided
            if durations is None:
                # Main sequence relationship: duration increases with amplitude
                duration = 0.02 + 0.002 * abs(amplitude)  # Base 20ms + 2ms per degree
            else:
                duration = durations[i]
                
            # Calculate peak velocity using main sequence if not provided
            if peak_velocities is None:
                # Main sequence relationship: peak velocity increases with amplitude
                peak_velocity = 200 + 20 * abs(amplitude)  # Base 200°/s + 20°/s per degree
            else:
                peak_velocity = peak_velocities[i]
            
            # Generate time points for this saccade
            n_samples = int(duration * self.sampling_rate)
            t = np.linspace(0, duration, n_samples)
            
            # Generate asymmetric velocity profile (faster acceleration than deceleration)
            velocity = peak_velocity * (np.exp(-((t - duration/3)**2) / (0.2 * duration)**2))
            
            # Integrate velocity to get position
            position = np.cumsum(velocity) / self.sampling_rate
            # Normalize to desired amplitude
            position = amplitude * position / np.max(position)
            
            # Add to signal at current time
            start_idx = int(current_time * self.sampling_rate)
            if start_idx + len(position) <= self.n_samples:
                if direction == 'vertical':
                    # Invert signal for upward movements
                    position = position if amplitude > 0 else -position
                signal[start_idx:start_idx + len(position)] += position
                
            # Update current time, adding gap between saccades
            current_time += duration + 0.05  # 50ms minimum gap
            
        return signal
    
    def simulate_smooth_pursuit(self, pattern: str = 'linear',
                              amplitude: float = 10.0,
                              frequency: float = 0.5,
                              direction: str = 'horizontal',
                              custom_trajectory: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate smooth pursuit eye movements with various tracking patterns.
        
        Args:
            pattern (str): Movement pattern type:
                - 'linear': Constant velocity back and forth
                - 'sinusoidal': Sinusoidal motion
                - 'circular': Circular motion
                - 'custom': Use provided custom_trajectory
            amplitude (float): Movement amplitude in degrees
            frequency (float): Movement frequency in Hz
            direction (str): Primary movement direction ('horizontal' or 'vertical')
                           Only used for linear and sinusoidal patterns
            custom_trajectory (Optional[np.ndarray]): Custom position trajectory
                                                    Used when pattern='custom'
                
        Returns:
            np.ndarray: Generated EOG signal containing smooth pursuit movements
            
        Notes:
            - Linear pattern: Constant velocity movement with sharp reversals
            - Sinusoidal pattern: Smooth sinusoidal motion
            - Circular pattern: Combination of horizontal and vertical sinusoids
            - Custom pattern: Allows arbitrary movement trajectories
            - Includes catch-up saccades for linear pattern to simulate pursuit lag
        """
        signal = np.zeros(self.n_samples)
        t = np.linspace(0, self.duration, self.n_samples)
        
        if pattern == 'linear':
            # Generate sawtooth wave for constant velocity
            period = 1.0 / frequency
            phase = 2 * np.pi * frequency * t
            position = amplitude * (2 * (phase % (2 * np.pi))/(2 * np.pi) - 1)
            
        elif pattern == 'sinusoidal':
            # Smooth sinusoidal motion
            position = amplitude * np.sin(2 * np.pi * frequency * t)
            
        elif pattern == 'circular':
            # Combine horizontal and vertical components
            horizontal = amplitude * np.cos(2 * np.pi * frequency * t)
            vertical = amplitude * np.sin(2 * np.pi * frequency * t)
            position = horizontal if direction == 'horizontal' else vertical
            
        elif pattern == 'custom' and custom_trajectory is not None:
            # Resample custom trajectory to match signal length
            t_custom = np.linspace(0, 1, len(custom_trajectory))
            t_signal = np.linspace(0, 1, self.n_samples)
            position = np.interp(t_signal, t_custom, custom_trajectory)
            
        else:
            raise ValueError(f"Unsupported pattern type: {pattern}")
            
        # Add catch-up saccades for linear pattern (pursuit lag)
        if pattern == 'linear':
            # Add small corrective saccades at direction changes
            period_samples = int(self.sampling_rate / frequency)
            for i in range(0, self.n_samples - period_samples, period_samples):
                # Add catch-up saccade
                saccade_amp = 0.1 * amplitude  # 10% of amplitude
                saccade_dur = 0.02  # 20ms
                saccade = self.simulate_saccades(
                    amplitudes=[saccade_amp],
                    directions=[direction],
                    durations=[saccade_dur]
                )
                saccade_start = i + period_samples - len(saccade)
                if saccade_start + len(saccade) <= self.n_samples:
                    signal[saccade_start:saccade_start + len(saccade)] += saccade
        
        # Add pursuit signal
        signal += position
        
        return signal
    
    def simulate_fixations(self, duration: float = 1.0,
                         microsaccade_rate: float = 2.0,
                         microsaccade_amplitude: float = 0.2,
                         drift_amplitude: float = 0.5,
                         tremor_amplitude: float = 0.1) -> np.ndarray:
        """Generate fixational eye movements including microsaccades, drift, and tremor.
        
        Args:
            duration (float): Duration of fixation period in seconds
            microsaccade_rate (float): Average number of microsaccades per second
            microsaccade_amplitude (float): Amplitude of microsaccades in degrees
            drift_amplitude (float): Maximum amplitude of slow drift in degrees
            tremor_amplitude (float): Amplitude of high-frequency tremor in degrees
                
        Returns:
            np.ndarray: Generated EOG signal containing fixational eye movements
            
        Notes:
            - Microsaccades: Small, rapid eye movements (typically 0.1-0.3°)
            - Drift: Slow, random motion between microsaccades
            - Tremor: High-frequency oscillations superimposed on drift
            - All components combine to maintain stable fixation
        """
        signal = np.zeros(self.n_samples)
        t = np.linspace(0, duration, self.n_samples)
        
        # Generate slow drift using random walk
        drift = np.zeros(self.n_samples)
        drift_velocity = np.random.normal(0, drift_amplitude/duration, self.n_samples)
        drift = np.cumsum(drift_velocity) / self.sampling_rate
        
        # Add high-frequency tremor
        tremor_freq = 80.0  # Hz
        tremor = tremor_amplitude * np.sin(2 * np.pi * tremor_freq * t)
        tremor += 0.5 * tremor_amplitude * np.sin(2 * np.pi * 2 * tremor_freq * t)
        
        # Add microsaccades at random intervals
        n_microsaccades = int(duration * microsaccade_rate)
        for _ in range(n_microsaccades):
            # Random timing
            start_time = np.random.uniform(0, duration - 0.02)  # 20ms minimum from end
            # Random amplitude and direction
            amplitude = np.random.uniform(-microsaccade_amplitude, microsaccade_amplitude)
            
            # Generate microsaccade using main sequence relationships
            microsaccade = self.simulate_saccades(
                amplitudes=[amplitude],
                directions=['horizontal'],
                durations=[0.02]  # Fixed 20ms duration for microsaccades
            )
            
            # Add to signal
            start_idx = int(start_time * self.sampling_rate)
            if start_idx + len(microsaccade) <= self.n_samples:
                signal[start_idx:start_idx + len(microsaccade)] += microsaccade
        
        # Combine all components
        signal += drift + tremor
        
        return signal
        
    def simulate_blinks(self, n_blinks: int = 3,
                       blink_duration: float = 0.2,
                       amplitude_range: Tuple[float, float] = (0.8, 1.2),
                       min_interval: float = 0.5,
                       natural_variability: bool = True) -> np.ndarray:
        """Generate natural eye blink patterns.
        
        Args:
            n_blinks (int): Number of blinks to generate
            blink_duration (float): Average duration of each blink in seconds
            amplitude_range (Tuple[float, float]): Range of blink amplitudes (min, max)
            min_interval (float): Minimum time between blinks in seconds
            natural_variability (bool): Add natural variations in blink shape and timing
                
        Returns:
            np.ndarray: Generated EOG signal containing blink artifacts
            
        Notes:
            - Implements asymmetric blink profile (faster closing than opening)
            - Adds natural variability in duration and amplitude
            - Ensures realistic minimum intervals between blinks
            - Models both full and partial blinks
        """
        signal = np.zeros(self.n_samples)
        available_time = self.duration - n_blinks * blink_duration
        if available_time < (n_blinks - 1) * min_interval:
            raise ValueError("Duration too short for specified number of blinks and intervals")
            
        # Generate blink timings with minimum intervals
        blink_times = []
        for _ in range(n_blinks):
            while True:
                time = np.random.uniform(0, self.duration - blink_duration)
                # Check if this time maintains minimum interval with existing blinks
                valid = True
                for existing_time in blink_times:
                    if abs(time - existing_time) < min_interval:
                        valid = False
                        break
                if valid:
                    blink_times.append(time)
                    break
        
        blink_times.sort()  # Ensure chronological order
        
        for start_time in blink_times:
            # Add natural variability if enabled
            if natural_variability:
                # Vary duration by ±20%
                dur = blink_duration * np.random.uniform(0.8, 1.2)
                # Vary amplitude within specified range
                amp = np.random.uniform(*amplitude_range)
                # Random chance of partial blink
                if np.random.random() < 0.2:  # 20% chance of partial blink
                    amp *= np.random.uniform(0.3, 0.7)
            else:
                dur = blink_duration
                amp = amplitude_range[1]
            
            # Generate asymmetric blink profile
            n_samples = int(dur * self.sampling_rate)
            t = np.linspace(-dur/2, dur/2, n_samples)
            
            # Faster closing (1/3 of duration) than opening (2/3 of duration)
            closing_mask = t < 0
            opening_mask = t >= 0
            
            blink = np.zeros(n_samples)
            blink[closing_mask] = amp * np.exp(-100 * (t[closing_mask]/(dur/6))**2)
            blink[opening_mask] = amp * np.exp(-50 * (t[opening_mask]/(dur/3))**2)
            
            # Add to signal
            start_idx = int(start_time * self.sampling_rate)
            if start_idx + len(blink) <= self.n_samples:
                signal[start_idx:start_idx + len(blink)] += blink
        
        return signal
    
    def generate(self, movement_type: str = 'saccades',
                pattern: str = 'horizontal',
                amplitude: float = 10.0,
                frequency: float = 0.5,
                n_saccades: int = 5,
                add_blinks: bool = False,
                n_blinks: int = 3,
                **kwargs) -> np.ndarray:
        """
        Generate synthetic EOG signal with specified eye movement pattern.
        
        Args:
            movement_type (str): Type of eye movement to generate:
                - 'saccades': Rapid eye movements
                - 'pursuit': Smooth pursuit movements
                - 'fixation': Stable gaze with microsaccades
            pattern (str): Movement pattern for pursuit ('linear', 'sinusoidal', 'circular')
                         or saccade direction ('horizontal', 'vertical')
            amplitude (float): Movement amplitude in degrees
            frequency (float): Movement frequency in Hz (for pursuit)
            n_saccades (int): Number of saccades to generate (for saccades)
            add_blinks (bool): Whether to add natural blink artifacts
            n_blinks (int): Number of blinks to add if add_blinks is True
            **kwargs: Additional parameters passed to specific simulation methods
            
        Returns:
            np.ndarray: Generated EOG signal
            
        Example:
            ```python
            # Generate smooth pursuit with blinks
            signal = eog_sim.generate(
                movement_type='pursuit',
                pattern='sinusoidal',
                amplitude=15,
                frequency=0.5,
                add_blinks=True,
                n_blinks=2
            )
            ```
        """
        signal = np.zeros(self.n_samples)
        
        if movement_type == 'saccades':
            # Generate random saccades
            amplitudes = [np.random.uniform(-amplitude, amplitude) for _ in range(n_saccades)]
            directions = [pattern] * n_saccades
            signal = self.simulate_saccades(amplitudes=amplitudes, directions=directions)
            
        elif movement_type == 'pursuit':
            # Generate smooth pursuit movement
            signal = self.simulate_smooth_pursuit(
                pattern=pattern,
                amplitude=amplitude,
                frequency=frequency,
                direction='horizontal',
                **kwargs
            )
            
        elif movement_type == 'fixation':
            # Generate fixation with microsaccades and drift
            signal = self.simulate_fixations(
                duration=self.duration,
                microsaccade_rate=kwargs.get('microsaccade_rate', 2.0),
                microsaccade_amplitude=kwargs.get('microsaccade_amplitude', 0.2),
                drift_amplitude=kwargs.get('drift_amplitude', 0.5),
                tremor_amplitude=kwargs.get('tremor_amplitude', 0.1)
            )
            
        else:
            raise ValueError(f"Unsupported movement type: {movement_type}")
            
        # Add blinks if requested
        if add_blinks:
            blinks = self.simulate_blinks(
                n_blinks=n_blinks,
                blink_duration=kwargs.get('blink_duration', 0.2),
                amplitude_range=kwargs.get('blink_amplitude_range', (0.8, 1.2)),
                min_interval=kwargs.get('min_blink_interval', 0.5),
                natural_variability=kwargs.get('natural_blink_variability', True)
            )
            signal += blinks
                start_time = np.random.uniform(0, self.duration - 0.3)
                blink = self.generate_blink()
                start_idx = int(start_time * self.sampling_rate)
                if start_idx + len(blink) <= self.n_samples:
                    signal[start_idx:start_idx + len(blink)] += blink
                    
        return signal