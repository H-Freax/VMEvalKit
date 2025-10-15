#!/usr/bin/env python3
"""
Veo 3.1 Debug Script - Direct test of Veo 3.1 API without the inference runner.

This bypasses the inference runner to test the WaveSpeed Veo 3.1 API directly.
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from vmevalkit.models.wavespeed_inference import WaveSpeedService, WaveSpeedModel, Veo31Service


def check_environment():
    """Check if environment is properly configured."""
    print("🔍 Environment Check:")
    
    api_key = os.getenv("WAVESPEED_API_KEY")
    if api_key:
        print(f"   ✅ WAVESPEED_API_KEY: SET (length: {len(api_key)})")
    else:
        print("   ❌ WAVESPEED_API_KEY: NOT SET")
        return False
    
    return True


async def test_veo31_direct():
    """Test Veo 3.1 directly using WaveSpeedService."""
    print("\n🎬 Testing Veo 3.1 Direct API Call")
    print("="*60)
    
    # Image path for testing
    image_path = Path("data/questions/maze_task/maze_0000/first_frame.png")
    if not image_path.exists():
        print(f"❌ Test image not found: {image_path}")
        return
    
    print(f"📁 Using image: {image_path}")
    print(f"📏 Image exists: {image_path.exists()}")
    print(f"📐 Image size: {image_path.stat().st_size} bytes")
    
    prompt = "Move the green dot from its starting position through the maze paths to the red flag. Navigate only through open spaces (white)."
    print(f"📝 Prompt: {prompt[:60]}...")
    
    # Test with WaveSpeedService using Veo 3.1 model
    try:
        print("\n🚀 Initializing WaveSpeedService with Veo 3.1...")
        service = WaveSpeedService(model=WaveSpeedModel.VEO_3_1_I2V)
        print(f"   Model: {service.model}")
        print(f"   Base URL: {service.base_url}")
        
        print("\n📤 Submitting generation request...")
        start_time = datetime.now()
        
        result = await service.generate_video(
            prompt=prompt,
            image_path=image_path,
            duration=8.0,
            resolution="720p",
            generate_audio=True,
            poll_timeout_s=30.0,  # Shorter timeout for testing
            poll_interval_s=2.0
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n✅ Request completed in {duration:.1f}s")
        print(f"📊 Result keys: {list(result.keys())}")
        print(f"🆔 Request ID: {result.get('request_id', 'N/A')}")
        print(f"🔗 Video URL: {result.get('video_url', 'N/A')}")
        
        return result
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n❌ Request failed after {duration:.1f}s")
        print(f"Error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        # Print more detailed error info
        import traceback
        print(f"\nFull traceback:")
        traceback.print_exc()
        
        return None


async def test_veo31_service():
    """Test using the convenience Veo31Service class."""
    print("\n🎬 Testing Veo 3.1 Service Class")
    print("="*60)
    
    # Image path for testing
    image_path = Path("data/questions/maze_task/maze_0000/first_frame.png")
    if not image_path.exists():
        print(f"❌ Test image not found: {image_path}")
        return
    
    prompt = "Move the green dot from its starting position through the maze paths to the red flag. Navigate only through open spaces (white)."
    
    try:
        print("🚀 Initializing Veo31Service...")
        service = Veo31Service()
        print(f"   Model: {service.model}")
        
        print("\n📤 Submitting generation request...")
        start_time = datetime.now()
        
        result = await service.generate_video(
            prompt=prompt,
            image_path=image_path,
            duration=8.0,
            resolution="720p",
            generate_audio=True,
            poll_timeout_s=30.0,  # Shorter timeout for testing
            poll_interval_s=2.0
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n✅ Request completed in {duration:.1f}s")
        print(f"📊 Result keys: {list(result.keys())}")
        print(f"🆔 Request ID: {result.get('request_id', 'N/A')}")
        print(f"🔗 Video URL: {result.get('video_url', 'N/A')}")
        
        return result
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n❌ Request failed after {duration:.1f}s")
        print(f"Error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        # Print more detailed error info
        import traceback
        print(f"\nFull traceback:")
        traceback.print_exc()
        
        return None


def main():
    """Main execution."""
    print("🔬 VEO 3.1 DEBUG SCRIPT")
    print("="*60)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment not properly configured. Please check your .env file.")
        return
    
    print("\n🎯 Testing Veo 3.1 API integration...")
    
    # Test both approaches
    print("\n" + "="*60)
    print("TEST 1: WaveSpeedService with Veo 3.1 Model")
    result1 = asyncio.run(test_veo31_direct())
    
    print("\n" + "="*60)
    print("TEST 2: Veo31Service Convenience Class")
    result2 = asyncio.run(test_veo31_service())
    
    # Summary
    print("\n" + "="*60)
    print("🎯 DEBUG SUMMARY")
    print("="*60)
    print(f"Test 1 (WaveSpeedService): {'✅ SUCCESS' if result1 else '❌ FAILED'}")
    print(f"Test 2 (Veo31Service): {'✅ SUCCESS' if result2 else '❌ FAILED'}")
    
    if result1 or result2:
        print(f"\n🎉 At least one test succeeded! Veo 3.1 API is working.")
    else:
        print(f"\n💥 Both tests failed. There may be an API configuration issue.")


if __name__ == "__main__":
    main()
