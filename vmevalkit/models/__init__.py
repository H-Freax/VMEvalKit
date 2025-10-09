"""
Video generation models for VMEvalKit.
"""

from .luma_inference import LumaInference, generate_video as luma_generate
from .veo_inference import VeoService
from .wavespeed_inference import WaveSpeedService
from .runway_inference import RunwayService
from .openai_inference import SoraService

# Open-source models (submodules)
from .ltx_inference import LTXVideoService, LTXVideoWrapper
from .hunyuan_inference import HunyuanVideoService, HunyuanVideoWrapper
from .videocrafter_inference import VideoCrafterService, VideoCrafterWrapper
from .dynamicrafter_inference import DynamiCrafterService, DynamiCrafterWrapper

__all__ = [
    # Commercial API services
    "LumaInference", "luma_generate", 
    "VeoService", "WaveSpeedService", "RunwayService", "SoraService",
    
    # Open-source models
    "LTXVideoService", "LTXVideoWrapper",
    "HunyuanVideoService", "HunyuanVideoWrapper", 
    "VideoCrafterService", "VideoCrafterWrapper",
    "DynamiCrafterService", "DynamiCrafterWrapper"
]
