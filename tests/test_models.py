import pytest
import numpy as np
import os
import tempfile
from models import ModelArtifact, ModelRegistry
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

@pytest.fixture
def sample_model():
    # Create a simple random forest classifier
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    return clf, X, y

@pytest.fixture
def sample_preprocessor():
    X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

def test_model_artifact_creation(sample_model):
    clf, _, _ = sample_model
    artifact = ModelArtifact(
        name="test_model",
        model_obj=clf,
        metadata={"version": "1.0"}
    )
    
    assert artifact.name == "test_model"
    assert artifact.metadata == {"version": "1.0"}
    assert artifact.preprocessor is None

def test_model_prediction(sample_model):
    clf, X, _ = sample_model
    artifact = ModelArtifact(
        name="test_model",
        model_obj=clf
    )
    
    # Test basic prediction
    pred = artifact.predict(X)
    assert isinstance(pred, np.ndarray)
    assert len(pred) == len(X)
    
    # Test probability prediction
    prob = artifact.predict_proba(X)
    assert isinstance(prob, np.ndarray)
    assert prob.shape == (len(X), 2)  # Binary classification

def test_model_with_preprocessor(sample_model, sample_preprocessor):
    clf, X, _ = sample_model
    artifact = ModelArtifact(
        name="test_model",
        model_obj=clf,
        preprocessor=sample_preprocessor
    )
    
    # Test prediction with preprocessing
    pred = artifact.predict(X)
    assert isinstance(pred, np.ndarray)
    assert len(pred) == len(X)

def test_model_save_load():
    # Create temporary directory for model storage
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save model
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        
        artifact = ModelArtifact(
            name="test_model",
            model_obj=clf,
            metadata={"version": "1.0"}
        )
        
        # Save model
        artifact.save(tmpdir)
        
        # Check saved files exist
        assert os.path.exists(os.path.join(tmpdir, "model.joblib"))
        assert os.path.exists(os.path.join(tmpdir, "meta.json"))
        
        # Load model
        loaded = ModelArtifact.load(tmpdir)
        assert loaded.name == "test_model"
        assert loaded.metadata == {"version": "1.0"}
        
        # Verify predictions match
        np.testing.assert_array_equal(
            artifact.predict(X),
            loaded.predict(X)
        )

def test_model_save_load_with_preprocessor(sample_preprocessor):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save model with preprocessor
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(sample_preprocessor.transform(X), y)
        
        artifact = ModelArtifact(
            name="test_model",
            model_obj=clf,
            preprocessor=sample_preprocessor,
            metadata={"version": "1.0"}
        )
        
        # Save model
        artifact.save(tmpdir)
        
        # Check preprocessor was saved
        assert os.path.exists(os.path.join(tmpdir, "preprocessor.joblib"))
        
        # Load model
        loaded = ModelArtifact.load(tmpdir)
        
        # Verify predictions match
        np.testing.assert_array_equal(
            artifact.predict(X),
            loaded.predict(X)
        )

def test_model_registry():
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(root=tmpdir)
        
        # Create test model
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        
        artifact = ModelArtifact(
            name="test_model",
            model_obj=clf,
            metadata={"version": "1.0"}
        )
        
        # Register model
        path = registry.register(artifact, version="v1")
        assert os.path.exists(path)
        
        # Check registry contents
        reg_contents = registry.list()
        assert "test_model:v1" in reg_contents
        assert reg_contents["test_model:v1"]["metadata"] == {"version": "1.0"}
        
        # Register another version
        path2 = registry.register(artifact, version="v2")
        assert os.path.exists(path2)
        assert "test_model:v2" in registry.list()

def test_model_without_proba():
    # Test model that doesn't support predict_proba
    class SimpleModel:
        def predict(self, X):
            return np.zeros(len(X))
    
    artifact = ModelArtifact(
        name="simple_model",
        model_obj=SimpleModel()
    )
    
    X = np.random.randn(10, 5)
    with pytest.raises(NotImplementedError):
        artifact.predict_proba(X)

def test_registry_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(root=tmpdir)
        assert registry.list() == {}