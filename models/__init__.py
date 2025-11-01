"""
Models module for machine learning model training and inference.
Provides base classes and implementations for various ML models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn

class BaseModel(ABC):
    """Base class for all models providing common functionality."""
    
    def __init__(self):
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on given data.
        
        Args:
            X: Training features
            y: Training labels
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Model predictions
        """
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance using multiple metrics.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted')
        }
    
    def get_confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Confusion matrix
        """
        y_pred = self.predict(X)
        return confusion_matrix(y, y_pred)
    
    def get_roc_curve(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate ROC curve points.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        y_pred = self.predict(X)
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        return fpr, tpr, thresholds
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv: Number of folds
            
        Returns:
            Dictionary with cross-validation scores
        """
        scores = cross_val_score(self.model, X, y, cv=cv)
        return {
            'mean_cv_score': scores.mean(),
            'std_cv_score': scores.std()
        }
    
    def save(self, filepath: str) -> None:
        """Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filepath: str) -> None:
        """Load model from disk.
        
        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
            self.is_fitted = True

class ClassicalModel(BaseModel):
    """Base class for classical ML models."""
    
    def __init__(self, model: Any):
        super().__init__()
        self.model = model
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)

class SVMModel(ClassicalModel):
    """Support Vector Machine implementation."""
    
    def __init__(self, **kwargs):
        from sklearn.svm import SVC
        super().__init__(SVC(**kwargs))

class RandomForestModel(ClassicalModel):
    """Random Forest implementation."""
    
    def __init__(self, **kwargs):
        from sklearn.ensemble import RandomForestClassifier
        super().__init__(RandomForestClassifier(**kwargs))

class DeepModel(BaseModel):
    """Base class for deep learning models."""
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Implementation details would depend on specific architecture
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            outputs = self.model(X_tensor)
            predictions = outputs.argmax(dim=1).cpu().numpy()
        return predictions

class CNNModel(DeepModel):
    """Convolutional Neural Network implementation."""
    
    def __init__(self, input_shape: Tuple[int, ...], num_classes: int):
        class CNN(nn.Module):
            def __init__(self, input_shape, num_classes):
                super().__init__()
                self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
                self.fc1 = nn.Linear(64 * ((input_shape[1]-4) * (input_shape[2]-4)), 128)
                self.fc2 = nn.Linear(128, num_classes)
                
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
                
        super().__init__(CNN(input_shape, num_classes))

class LSTMModel(DeepModel):
    """Long Short-Term Memory Network implementation."""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        class LSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, num_classes)
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                out = self.fc(lstm_out[:, -1, :])
                return out
                
        super().__init__(LSTM(input_size, hidden_size, num_classes))

class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models."""
    
    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = models
        self.weights = weights if weights is not None else [1/len(models)] * len(models)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for model in self.models:
            model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        predictions = np.array([model.predict(X) for model in self.models])
        weighted_sum = np.average(predictions, axis=0, weights=self.weights)
        return np.round(weighted_sum).astype(int)