import os
import pickle
from datetime import datetime
from typing import Dict, Any, Optional

class ModelManager:
    """
    Manages the saving, loading, and versioning of trained machine learning models.
    """
    def __init__(self, base_save_dir: str = "models/saved_models"):
        self.base_save_dir = base_save_dir
        os.makedirs(self.base_save_dir, exist_ok=True)
        self.loaded_models: Dict[str, Any] = {} # Stores currently loaded models

    def _generate_model_id(self, model_name: str) -> str:
        """Generates a unique ID for a model based on its name and timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_name}_{timestamp}"

    def save_model(self, model: Any, model_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Saves a trained model to disk.
        
        Args:
            model: The trained machine learning model object.
            model_name: A descriptive name for the model (e.g., "SVM_Classifier").
            metadata: Optional dictionary of additional metadata to save with the model.
            
        Returns:
            The unique ID of the saved model.
        """
        model_id = self._generate_model_id(model_name)
        model_dir = os.path.join(self.base_save_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "model.pkl")
        metadata_path = os.path.join(model_dir, "metadata.pkl")

        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            full_metadata = {
                "model_id": model_id,
                "model_name": model_name,
                "save_time": datetime.now().isoformat(),
                "path": model_path,
                **(metadata if metadata is not None else {})
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(full_metadata, f)
            
            print(f"Model '{model_name}' saved successfully with ID: {model_id}")
            return model_id
        except Exception as e:
            print(f"Error saving model '{model_name}': {e}")
            raise

    def load_model(self, model_id: str) -> Any:
        """
        Loads a trained model from disk using its unique ID.
        
        Args:
            model_id: The unique ID of the model to load.
            
        Returns:
            The loaded machine learning model object.
        """
        if model_id in self.loaded_models:
            print(f"Model '{model_id}' already loaded from cache.")
            return self.loaded_models[model_id]

        model_dir = os.path.join(self.base_save_dir, model_id)
        model_path = os.path.join(model_dir, "model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found for ID: {model_id} at {model_path}")

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.loaded_models[model_id] = model
            print(f"Model '{model_id}' loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model '{model_id}': {e}")
            raise

    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves metadata for a specific model.
        """
        metadata_path = os.path.join(self.base_save_dir, model_id, "metadata.pkl")
        if not os.path.exists(metadata_path):
            return None
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            return metadata
        except Exception as e:
            print(f"Error reading metadata for model '{model_id}': {e}")
            return None

    def list_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Lists all saved models and their basic metadata.
        
        Returns:
            A dictionary where keys are model_ids and values are their metadata.
        """
        all_models = {}
        if not os.path.exists(self.base_save_dir):
            return all_models

        for model_id in os.listdir(self.base_save_dir):
            model_dir = os.path.join(self.base_save_dir, model_id)
            if os.path.isdir(model_dir):
                metadata = self.get_model_metadata(model_id)
                if metadata:
                    all_models[model_id] = metadata
        return all_models

    def deploy_model(self, model_id: str) -> Any:
        """
        Loads and "deploys" a model, making it ready for real-time inference.
        In a real application, this might involve setting up an API endpoint,
        moving the model to a production environment, or initializing a prediction service.
        For this UI, it primarily means loading it and marking it as active for inference.
        """
        print(f"Attempting to deploy model: {model_id}")
        try:
            model = self.load_model(model_id)
            # Here, you would add logic for actual deployment, e.g.,
            # self.active_inference_model = model
            # self.active_inference_model_id = model_id
            print(f"Model '{model_id}' successfully deployed for inference.")
            return model
        except Exception as e:
            print(f"Failed to deploy model '{model_id}': {e}")
            raise

# Example Usage (for testing purposes)
if __name__ == "__main__":
    from sklearn.svm import SVC
    from sklearn.datasets import make_classification

    # Create a dummy model
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    dummy_model = SVC(kernel='linear', C=1)
    dummy_model.fit(X, y)

    manager = ModelManager()

    # Save a model
    model_id_1 = manager.save_model(dummy_model, "SVM_Classifier_v1", {"dataset": "synthetic_data_v1"})
    print(f"Saved model with ID: {model_id_1}")

    # List all models
    print("\nAll saved models:")
    for mid, meta in manager.list_all_models().items():
        print(f"- ID: {mid}, Name: {meta.get('model_name')}, Saved: {meta.get('save_time')}")

    # Load a model
    loaded_model = manager.load_model(model_id_1)
    print(f"\nLoaded model type: {type(loaded_model)}")

    # Deploy a model
    deployed_model = manager.deploy_model(model_id_1)
    print(f"Deployed model type: {type(deployed_model)}")

    # Simulate another model
    dummy_model_2 = SVC(kernel='rbf', C=0.5)
    dummy_model_2.fit(X, y)
    model_id_2 = manager.save_model(dummy_model_2, "SVM_Classifier_v2", {"dataset": "synthetic_data_v2"})
    print(f"Saved model with ID: {model_id_2}")

    print("\nAll saved models after adding another:")
    for mid, meta in manager.list_all_models().items():
        print(f"- ID: {mid}, Name: {meta.get('model_name')}, Saved: {meta.get('save_time')}")