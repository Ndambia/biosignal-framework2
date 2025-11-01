"""
Biosignal acquisition module for EMG, ECG, and EOG signals.
Provides classes for loading and interfacing with various biosignals.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Dict
import numpy as np
import numpy.typing as npt
from pathlib import Path
import json
import h5py
import mne
import threading
import queue
import time

class BioSignalAcquisition(ABC):
    """Base class for biosignal acquisition."""
    
    def __init__(self, sampling_rate: float = 1000.0):
        """
        Initialize the biosignal acquisition.

        Args:
            sampling_rate: Sampling rate in Hz (default: 1000.0)
        """
        self.sampling_rate = sampling_rate
        self.data: Optional[npt.NDArray[np.float64]] = None
        self.timestamps: Optional[npt.NDArray[np.float64]] = None
        self.metadata: Dict = {}
        self._acquisition_thread: Optional[threading.Thread] = None
        self._data_queue: queue.Queue = queue.Queue()
        self._stop_flag = threading.Event()

    @abstractmethod
    def load_from_file(self, filepath: str) -> None:
        """
        Load signal data from a file.

        Args:
            filepath: Path to the signal data file
        """
        pass

    @abstractmethod
    def start_acquisition(self) -> None:
        """Start real-time signal acquisition."""
        pass

    @abstractmethod
    def stop_acquisition(self) -> None:
        """Stop real-time signal acquisition."""
        if self._acquisition_thread and self._acquisition_thread.is_alive():
            self._stop_flag.set()
            self._acquisition_thread.join()
            self._stop_flag.clear()

    def get_data(self) -> npt.NDArray[np.float64]:
        """
        Get the acquired signal data.

        Returns:
            NDArray containing the signal data
        """
        if self.data is None:
            raise ValueError("No data has been acquired yet")
        return self.data

    def get_metadata(self) -> Dict:
        """
        Get signal metadata.

        Returns:
            Dictionary containing signal metadata
        """
        return self.metadata

    def _load_hdf5(self, filepath: str) -> None:
        """
        Load data from HDF5 file format.
        
        Args:
            filepath: Path to HDF5 file
        """
        with h5py.File(filepath, 'r') as f:
            self.data = f['data'][:]
            self.timestamps = f['timestamps'][:]
            if 'metadata' in f:
                self.metadata.update(json.loads(f['metadata'][()]))

    def _load_numpy(self, filepath: str) -> None:
        """
        Load data from NumPy .npy or .npz format.
        
        Args:
            filepath: Path to NumPy file
        """
        if filepath.endswith('.npz'):
            data = np.load(filepath)
            self.data = data['data']
            self.timestamps = data['timestamps']
            if 'metadata' in data:
                self.metadata.update(data['metadata'].item())
        else:
            self.data = np.load(filepath)
            self.timestamps = np.arange(len(self.data)) / self.sampling_rate

class EMGAcquisition(BioSignalAcquisition):
    """Class for EMG signal acquisition."""

    def __init__(self, sampling_rate: float = 1000.0, channels: int = 1):
        """
        Initialize EMG acquisition.

        Args:
            sampling_rate: Sampling rate in Hz (default: 1000.0)
            channels: Number of EMG channels (default: 1)
        """
        super().__init__(sampling_rate)
        self.channels = channels
        self.metadata.update({"signal_type": "EMG", "channels": channels})
        self._buffer_size = int(sampling_rate * 1)  # 1 second buffer

    def load_from_file(self, filepath: str) -> None:
        """
        Load EMG data from a file.

        Args:
            filepath: Path to the EMG data file
            
        Supported formats:
            - .hdf5/.h5: HDF5 format
            - .npy/.npz: NumPy format
            - .edf: European Data Format
            - .xdf: XDF format (LSL)
        """
        filepath = str(Path(filepath))
        
        if filepath.endswith(('.h5', '.hdf5')):
            self._load_hdf5(filepath)
        elif filepath.endswith(('.npy', '.npz')):
            self._load_numpy(filepath)
        elif filepath.endswith('.edf'):
            raw = mne.io.read_raw_edf(filepath, preload=True)
            self.data = raw.get_data()
            self.sampling_rate = raw.info['sfreq']
            self.timestamps = np.arange(len(self.data[0])) / self.sampling_rate
        else:
            raise ValueError(f"Unsupported file format for {filepath}")

    def _acquire_data(self) -> None:
        """Background thread function for real-time data acquisition."""
        buffer = np.zeros((self.channels, self._buffer_size))
        sample_interval = 1.0 / self.sampling_rate
        
        while not self._stop_flag.is_set():
            # Simulate EMG acquisition - replace with actual hardware interface
            new_sample = np.random.normal(0, 1, (self.channels, 1))
            buffer = np.roll(buffer, -1, axis=1)
            buffer[:, -1:] = new_sample
            
            self._data_queue.put(new_sample)
            time.sleep(sample_interval)

    def start_acquisition(self) -> None:
        """Start real-time EMG acquisition."""
        if self._acquisition_thread and self._acquisition_thread.is_alive():
            raise RuntimeError("Acquisition already running")
            
        self._acquisition_thread = threading.Thread(target=self._acquire_data)
        self._acquisition_thread.daemon = True
        self._acquisition_thread.start()

    def stop_acquisition(self) -> None:
        """Stop real-time EMG acquisition."""
        super().stop_acquisition()

class ECGAcquisition(BioSignalAcquisition):
    """Class for ECG signal acquisition."""

    def __init__(self, sampling_rate: float = 250.0, leads: List[str] = None):
        """
        Initialize ECG acquisition.

        Args:
            sampling_rate: Sampling rate in Hz (default: 250.0)
            leads: List of ECG leads to acquire (default: ["II"])
        """
        super().__init__(sampling_rate)
        self.leads = leads if leads is not None else ["II"]
        self.metadata.update({"signal_type": "ECG", "leads": self.leads})
        self._buffer_size = int(sampling_rate * 2)  # 2 second buffer

    def load_from_file(self, filepath: str) -> None:
        """
        Load ECG data from a file.

        Args:
            filepath: Path to the ECG data file
            
        Supported formats:
            - .hdf5/.h5: HDF5 format
            - .npy/.npz: NumPy format
            - .edf: European Data Format
            - .wfdb: WFDB format (PhysioNet)
        """
        filepath = str(Path(filepath))
        
        if filepath.endswith(('.h5', '.hdf5')):
            self._load_hdf5(filepath)
        elif filepath.endswith(('.npy', '.npz')):
            self._load_numpy(filepath)
        elif filepath.endswith('.edf'):
            raw = mne.io.read_raw_edf(filepath, preload=True)
            self.data = raw.get_data()
            self.sampling_rate = raw.info['sfreq']
            self.timestamps = np.arange(len(self.data[0])) / self.sampling_rate
        elif filepath.endswith('.wfdb'):
            import wfdb
            record = wfdb.rdrecord(filepath.replace('.wfdb', ''))
            self.data = record.p_signal.T
            self.sampling_rate = record.fs
            self.timestamps = np.arange(len(self.data[0])) / self.sampling_rate
            self.metadata.update({"leads": record.sig_name})
        else:
            raise ValueError(f"Unsupported file format for {filepath}")

    def _acquire_data(self) -> None:
        """Background thread function for real-time data acquisition."""
        buffer = np.zeros((len(self.leads), self._buffer_size))
        sample_interval = 1.0 / self.sampling_rate
        
        while not self._stop_flag.is_set():
            # Simulate ECG acquisition - replace with actual hardware interface
            new_sample = np.random.normal(0, 0.1, (len(self.leads), 1))
            buffer = np.roll(buffer, -1, axis=1)
            buffer[:, -1:] = new_sample
            
            self._data_queue.put(new_sample)
            time.sleep(sample_interval)

    def start_acquisition(self) -> None:
        """Start real-time ECG acquisition."""
        if self._acquisition_thread and self._acquisition_thread.is_alive():
            raise RuntimeError("Acquisition already running")
            
        self._acquisition_thread = threading.Thread(target=self._acquire_data)
        self._acquisition_thread.daemon = True
        self._acquisition_thread.start()

    def stop_acquisition(self) -> None:
        """Stop real-time ECG acquisition."""
        super().stop_acquisition()

class EOGAcquisition(BioSignalAcquisition):
    """Class for EOG signal acquisition."""

    def __init__(self, sampling_rate: float = 200.0, channels: List[str] = None):
        """
        Initialize EOG acquisition.

        Args:
            sampling_rate: Sampling rate in Hz (default: 200.0)
            channels: List of EOG channels to acquire (default: ["Horizontal", "Vertical"])
        """
        super().__init__(sampling_rate)
        self.channels = channels if channels is not None else ["Horizontal", "Vertical"]
        self.metadata.update({"signal_type": "EOG", "channels": self.channels})
        self._buffer_size = int(sampling_rate * 1)  # 1 second buffer

    def load_from_file(self, filepath: str) -> None:
        """
        Load EOG data from a file.

        Args:
            filepath: Path to the EOG data file
            
        Supported formats:
            - .hdf5/.h5: HDF5 format
            - .npy/.npz: NumPy format
            - .edf: European Data Format
            - .fif: Elekta Neuromag format
        """
        filepath = str(Path(filepath))
        
        if filepath.endswith(('.h5', '.hdf5')):
            self._load_hdf5(filepath)
        elif filepath.endswith(('.npy', '.npz')):
            self._load_numpy(filepath)
        elif filepath.endswith('.edf'):
            raw = mne.io.read_raw_edf(filepath, preload=True)
            self.data = raw.get_data()
            self.sampling_rate = raw.info['sfreq']
            self.timestamps = np.arange(len(self.data[0])) / self.sampling_rate
        elif filepath.endswith('.fif'):
            raw = mne.io.read_raw_fif(filepath, preload=True)
            self.data = raw.get_data()
            self.sampling_rate = raw.info['sfreq']
            self.timestamps = np.arange(len(self.data[0])) / self.sampling_rate
        else:
            raise ValueError(f"Unsupported file format for {filepath}")

    def _acquire_data(self) -> None:
        """Background thread function for real-time data acquisition."""
        buffer = np.zeros((len(self.channels), self._buffer_size))
        sample_interval = 1.0 / self.sampling_rate
        
        while not self._stop_flag.is_set():
            # Simulate EOG acquisition - replace with actual hardware interface
            new_sample = np.random.normal(0, 0.5, (len(self.channels), 1))
            buffer = np.roll(buffer, -1, axis=1)
            buffer[:, -1:] = new_sample
            
            self._data_queue.put(new_sample)
            time.sleep(sample_interval)

    def start_acquisition(self) -> None:
        """Start real-time EOG acquisition."""
        if self._acquisition_thread and self._acquisition_thread.is_alive():
            raise RuntimeError("Acquisition already running")
            
        self._acquisition_thread = threading.Thread(target=self._acquire_data)
        self._acquisition_thread.daemon = True
        self._acquisition_thread.start()

    def stop_acquisition(self) -> None:
        """Stop real-time EOG acquisition."""
        super().stop_acquisition()