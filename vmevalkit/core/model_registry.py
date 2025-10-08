"""
Model registry for managing video generation models.
"""

from typing import Dict, Any, Optional, Type
from ..models.base import BaseVideoModel
from ..api_clients.runway_client import RunwayModel
from ..api_clients.luma_client import LumaDreamMachine
from ..api_clients.veo_client import GoogleVeo


class ModelRegistry:
    """
    Registry for video generation models.
    
    IMPORTANT: Only models that support text+image→video are suitable for VMEvalKit.
    """
    
    # Registered model classes - All support text+image input
    _models: Dict[str, Type[BaseVideoModel]] = {
        "luma-dream-machine": LumaDreamMachine,
        "google-veo-001": GoogleVeo,
        "google-veo-002": GoogleVeo,
        "runway-gen4-aleph": RunwayModel,  # Supports text+image via workaround
    }
    
    # Model compatibility status - Only includes implemented models
    _compatibility = {
        "luma-dream-machine": {
            "supports_text_image": True,
            "status": "✅ Compatible",
            "notes": "Supports text prompts with image references"
        },
        "google-veo-001": {
            "supports_text_image": True,
            "status": "✅ Compatible",
            "notes": "Google Veo v1 - High-quality text+image→video generation via Vertex AI"
        },
        "google-veo-002": {
            "supports_text_image": True,
            "status": "✅ Compatible",
            "notes": "Google Veo v2 (latest) - Improved quality and consistency, text+image→video"
        },
        "runway-gen4-aleph": {
            "supports_text_image": True,
            "status": "✅ Compatible",
            "notes": "Video+text input - works by converting image to video first (automatic workaround)"
        }
    }
    
    @classmethod
    def load_model(
        cls,
        model_name: str,
        api_key: Optional[str] = None,
        device: str = "cuda",
        **kwargs
    ) -> BaseVideoModel:
        """
        Load a video generation model.
        
        Args:
            model_name: Name of the model to load
            api_key: API key for closed-source models
            device: Device for computation (for local models)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Initialized model instance
            
        Raises:
            ValueError: If model doesn't support required capabilities
        """
        # Check compatibility
        if model_name in cls._compatibility:
            compat = cls._compatibility[model_name]
            print(f"\nLoading model: {model_name}")
            print(f"Status: {compat['status']}")
            print(f"Notes: {compat['notes']}")
        
        # Handle Runway models specially
        if model_name.startswith("runway-"):
            # Extract model variant
            variant = model_name.replace("runway-", "").replace("-", "_")
            return RunwayModel(model_name=variant, api_key=api_key, **kwargs)
        
        # Handle Google Veo models
        if model_name.startswith("google-veo-"):
            # Extract version (001 or 002)
            version = model_name.replace("google-", "")
            return GoogleVeo(model_version=version, api_key=api_key, **kwargs)
        
        # Load other registered models
        if model_name in cls._models:
            model_class = cls._models[model_name]
            return model_class(api_key=api_key, **kwargs)
        
        # Model not found
        available = list(cls._models.keys())
        
        raise ValueError(
            f"Unknown model: {model_name}\n"
            f"Available models: {available}\n"
            f"All models support text+image→video generation."
        )
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseVideoModel]):
        """
        Register a custom model.
        
        Args:
            name: Model identifier
            model_class: Model class (must inherit from BaseVideoModel)
        """
        if not issubclass(model_class, BaseVideoModel):
            raise ValueError(
                f"Model class must inherit from BaseVideoModel"
            )
        
        cls._models[name] = model_class
        print(f"✅ Registered model: {name}")
    
    @classmethod
    def list_models(cls, only_compatible: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        List all registered models.
        
        Args:
            only_compatible: If True, only show models that support text+image
            
        Returns:
            Dictionary of model information
        """
        models = {}
        
        for name, compat in cls._compatibility.items():
            if only_compatible and not compat["supports_text_image"]:
                continue
            
            models[name] = compat
        
        return models
    
    @classmethod
    def check_compatibility(cls, model_name: str) -> bool:
        """
        Check if a model supports text+image→video.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if compatible, False otherwise
        """
        if model_name in cls._compatibility:
            return cls._compatibility[model_name]["supports_text_image"]
        return False
