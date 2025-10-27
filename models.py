"""Model manager, registry helpers and packaging utilities.

Implements:
- ModelWrapper: standardized load/predict/save
- ModelRegistry: simple local registry storing model metadata and paths
- Export helpers for ONNX/TFLite (stubs - require framework-specific code)
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import os
import json
import joblib

@dataclass
class ModelArtifact:
    name: str
    model_obj: Any
    preprocessor: Optional[Any] = None
    metadata: Dict[str, Any] = None

    def predict(self, X):
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)
        return self.model_obj.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model_obj, 'predict_proba'):
            if self.preprocessor is not None:
                X = self.preprocessor.transform(X)
            return self.model_obj.predict_proba(X)
        raise NotImplementedError

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, 'model.joblib')
        joblib.dump(self.model_obj, path)
        meta = {'name': self.name, 'metadata': self.metadata}
        if self.preprocessor is not None:
            joblib.dump(self.preprocessor, os.path.join(directory, 'preprocessor.joblib'))
        with open(os.path.join(directory, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> 'ModelArtifact':
        with open(os.path.join(directory, 'meta.json'), 'r') as f:
            meta = json.load(f)
        model = joblib.load(os.path.join(directory, 'model.joblib'))
        pre = None
        pre_path = os.path.join(directory, 'preprocessor.joblib')
        if os.path.exists(pre_path):
            pre = joblib.load(pre_path)
        return cls(name=meta.get('name', 'unknown'), model_obj=model, preprocessor=pre, metadata=meta.get('metadata', {}))

class ModelRegistry:
    def __init__(self, root: str = 'experiments/models'):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def register(self, artifact: ModelArtifact, version: str) -> str:
        path = os.path.join(self.root, f"{artifact.name}_{version}")
        artifact.save(path)
        # write registry entry
        reg_path = os.path.join(self.root, 'registry.json')
        registry = {}
        if os.path.exists(reg_path):
            with open(reg_path, 'r') as f:
                registry = json.load(f)
        registry[f"{artifact.name}:{version}"] = {'path': path, 'metadata': artifact.metadata}
        with open(reg_path, 'w') as f:
            json.dump(registry, f, indent=2)
        return path

    def list(self):
        reg_path = os.path.join(self.root, 'registry.json')
        if os.path.exists(reg_path):
            with open(reg_path, 'r') as f:
                return json.load(f)
        return {}
