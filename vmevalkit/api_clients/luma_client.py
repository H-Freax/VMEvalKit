"""
Luma Dream Machine API client implementation.

Luma Dream Machine supports text prompts with image references for video generation,
making it suitable for VMEvalKit's reasoning tasks.
"""

import os
import time
import base64
import requests
from typing import Optional, Dict, Any, Union
from pathlib import Path
import json
from io import BytesIO
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from ..models.base import BaseVideoModel
from ..utils.local_image_server import get_image_server


class LumaAPIError(Exception):
    """Custom exception for Luma API errors."""
    pass


class LumaDreamMachine(BaseVideoModel):
    """
    Luma Dream Machine API implementation.
    
    This model supports text+image→video generation required for reasoning tasks.
    """
    
    BASE_URL = "https://api.lumalabs.ai/dream-machine/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        enhance_prompt: bool = True,
        loop_video: bool = False,
        aspect_ratio: str = "16:9",
        model: str = "ray-2",
        verbose: bool = True,
        **kwargs
    ):
        """
        Initialize Luma Dream Machine client.
        
        Args:
            api_key: Luma API key (or set LUMA_API_KEY env var)
            enhance_prompt: Whether to use Luma's prompt enhancement
            loop_video: Whether to create looping videos
            aspect_ratio: Output aspect ratio
            verbose: Whether to show detailed progress updates
        """
        self.api_key = api_key or os.getenv("LUMA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Luma API key required. Set LUMA_API_KEY env var or pass api_key parameter."
            )
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.enhance_prompt = enhance_prompt
        self.loop_video = loop_video
        self.aspect_ratio = aspect_ratio
        self.model = model
        self.verbose = verbose
        
        # Print debug info
        if self.verbose:
            print(f"[Luma] Initialized with enhance_prompt={enhance_prompt}, model={model}")
        
        super().__init__(name="luma_dream_machine", **kwargs)
    
    def supports_text_image_input(self) -> bool:
        """
        Luma Dream Machine supports both text and image inputs.
        
        Returns:
            True - Luma accepts text prompts with image references
        """
        return True
    
    def generate(
        self,
        image: Union[str, Path, Image.Image],
        text_prompt: str,
        duration: float = 5.0,
        fps: int = 24,
        resolution: tuple = (1280, 720),
        **kwargs
    ) -> str:
        """
        Generate video from text prompt and image using Luma Dream Machine.
        
        Args:
            image: Input image for reference
            text_prompt: Text instructions for video generation
            duration: Video duration in seconds
            fps: Frames per second (Luma may override this)
            resolution: Output resolution
            **kwargs: Additional parameters
            
        Returns:
            Path to generated video file
        """
        # Ensure public image URL per Luma docs (keyframes requires URL)
        image_url = self._ensure_image_url(image)

        # Map resolution to Luma's accepted strings
        _, height = resolution
        if height >= 2160:
            resolution_str = "4k"
        elif height >= 1080:
            resolution_str = "1080"
        elif height >= 720:
            resolution_str = "720p"
        else:
            resolution_str = "540p"

        payload = {
            "prompt": text_prompt,
            "model": self.model,
            "keyframes": {
                "frame0": {
                    "type": "image",
                    "url": image_url
                }
            },
            "enhance_prompt": self.enhance_prompt,
            "loop": self.loop_video,
            "aspect_ratio": self.aspect_ratio,
            "duration": f"{int(duration)}s",
            "resolution": resolution_str
        }
        
        print(f"[Luma] Sending request with:")
        print(f"  - Prompt: {text_prompt}")
        print(f"  - Image URL: {image_url}")
        print(f"  - Model: {self.model}")
        print(f"  - Enhance prompt: {self.enhance_prompt}")

        # Create generation request
        generation_id = self._start_generation(payload=payload)
        
        # Poll for completion
        video_url = self._poll_for_completion(generation_id)
        
        # Download video
        video_path = self._download_video(video_url, generation_id)
        
        return video_path
    
    def _encode_image(self, image: Image.Image) -> bytes:
        """Encode PIL Image to raw PNG bytes for upload."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return buffered.getvalue()

    def _ensure_image_url(self, image: Union[str, Path, Image.Image]) -> str:
        """
        Ensure we have a public HTTP URL for the image.
        
        According to Luma docs:
        "You should upload and use your own cdn image urls, 
        currently this is the only way to pass an image"
        """
        if isinstance(image, str) and (image.startswith("http://") or image.startswith("https://")):
            # Already a public URL
            return image
        
        # For local images, we need to serve them via HTTP
        # Start local server if needed
        server = get_image_server(directory=os.getcwd())
        
        # Save image to a temporary file if it's a PIL Image
        if isinstance(image, Image.Image):
            temp_path = Path("temp_image.png")
            image.save(temp_path)
            image = temp_path
        
        # Convert to Path object
        image_path = Path(image)
        if not image_path.is_absolute():
            image_path = Path(os.getcwd()) / image_path
        
        # Get HTTP URL from local server
        url = server.get_url(str(image_path))
        
        print(f"[Luma] Serving local image via HTTP: {url}")
        return url
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def _start_generation(
        self,
        payload: Dict[str, Any],
    ) -> str:
        """
        Start video generation job.
        
        Returns:
            Generation ID for polling
        """
        endpoint = f"{self.BASE_URL}/generations"
        
        try:
            response = requests.post(endpoint, headers=self.headers, json=payload)
            # On error, surface response content for debugging
            if not response.ok:
                try:
                    err_body = response.text
                except Exception:
                    err_body = "<no body>"
                raise LumaAPIError(
                    f"Failed to start generation: HTTP {response.status_code} - {err_body}"
                )
            data = response.json()
            return data.get("id") or data.get("generation_id")
        except requests.exceptions.RequestException as e:
            raise LumaAPIError(f"Failed to start generation: {e}")
    
    def _poll_for_completion(
        self,
        generation_id: str,
        max_wait: int = 600,
        initial_poll_interval: int = 2
    ) -> str:
        """
        Poll for generation completion with exponential backoff.
        
        Returns:
            URL of generated video
        """
        endpoint = f"{self.BASE_URL}/generations/{generation_id}"
        
        start_time = time.time()
        poll_count = 0
        poll_interval = initial_poll_interval
        max_poll_interval = 30  # Cap at 30 seconds
        last_status = None
        
        if self.verbose:
            print(f"\n[Luma] Waiting for generation {generation_id} to complete...")
            print(f"[Luma] This typically takes 2-4 minutes depending on server load.")
        
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(endpoint, headers=self.headers)
                response.raise_for_status()
                data = response.json()
                
                status = data.get("state")
                elapsed = int(time.time() - start_time)
                
                # Show progress update if status changed or periodically
                if status != last_status or (self.verbose and poll_count % 3 == 0):
                    # Format elapsed time
                    mins, secs = divmod(elapsed, 60)
                    time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
                    
                    if self.verbose:
                        print(f"[Luma] Status: {status} | Elapsed: {time_str} | Next check in {poll_interval}s")
                    elif status != last_status:
                        print(f"[Luma] Status: {status} | Elapsed: {time_str}")
                    
                    last_status = status
                
                if status == "completed":
                    if self.verbose:
                        print(f"[Luma] ✓ Generation completed in {elapsed} seconds!")
                    
                    # Handle different response formats
                    video_url = None
                    if "video" in data and "url" in data["video"]:
                        video_url = data["video"]["url"]
                    elif "assets" in data and "video" in data["assets"]:
                        video_url = data["assets"]["video"]
                    elif "download_url" in data:
                        video_url = data["download_url"]
                    
                    if video_url:
                        return video_url
                    else:
                        # Debug: print the actual response structure
                        if self.verbose:
                            print(f"[Luma] DEBUG: Response structure: {list(data.keys())}")
                            if "assets" in data:
                                print(f"[Luma] DEBUG: Assets keys: {list(data['assets'].keys())}")
                        raise LumaAPIError(f"Could not find video URL in response: {data}")
                elif status == "failed":
                    error_msg = data.get("failure_reason", "Unknown error")
                    raise LumaAPIError(f"Generation failed: {error_msg}")
                
                # Still processing - sleep with exponential backoff
                time.sleep(poll_interval)
                poll_count += 1
                
                # Exponential backoff: start fast, then slow down
                if poll_count <= 5:
                    # First 5 polls: every 2 seconds
                    poll_interval = initial_poll_interval
                elif poll_count <= 10:
                    # Next 5 polls: every 5 seconds  
                    poll_interval = 5
                elif poll_count <= 20:
                    # Next 10 polls: every 10 seconds
                    poll_interval = 10
                else:
                    # After that: increase gradually up to max
                    poll_interval = min(poll_interval * 1.2, max_poll_interval)
                
            except requests.exceptions.RequestException as e:
                raise LumaAPIError(f"Failed to check status: {e}")
        
        raise LumaAPIError(f"Generation timed out after {max_wait} seconds")
    
    def _download_video(self, video_url: str, generation_id: str) -> str:
        """
        Download generated video.
        
        Returns:
            Path to downloaded video file
        """
        output_dir = Path("./outputs")
        output_dir.mkdir(exist_ok=True)
        
        video_path = output_dir / f"luma_{generation_id}.mp4"
        
        try:
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            
            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return str(video_path)
            
        except requests.exceptions.RequestException as e:
            raise LumaAPIError(f"Failed to download video: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.name,
            "supports_text_image": True,
            "api": "Luma Dream Machine",
            "capabilities": {
                "text_prompt": True,
                "image_reference": True,
                "max_duration": 10,  # seconds
                "aspect_ratios": ["16:9", "9:16", "1:1", "4:3", "3:4"],
                "enhance_prompt": self.enhance_prompt,
                "loop": self.loop_video
            }
        }
