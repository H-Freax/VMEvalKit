#!/usr/bin/env python3
"""
Unit tests for verified Luma models.
Each test uses one image+text pair and verifies the model works correctly.
"""

import unittest
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from vmevalkit import InferenceRunner
from vmevalkit.core.model_registry import ModelRegistry


class TestLumaModels(unittest.TestCase):
    """Test suite for Luma video generation models."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.runner = InferenceRunner(output_dir="./test_outputs")
        
        # Use a simple test image
        cls.test_image = "data/generated_mazes/irregular_0000_first.png"
        cls.test_prompt = "Show a path being drawn through this maze from start to finish"
        
        # Check if test image exists
        if not Path(cls.test_image).exists():
            raise FileNotFoundError(f"Test image not found: {cls.test_image}")
    
    def test_luma_dream_machine(self):
        """Test luma-dream-machine model."""
        model_name = "luma-dream-machine"
        
        try:
            # Check if model is registered
            self.assertIn(model_name, ModelRegistry.list_models())
            
            # Run inference
            result = self.runner.run(
                model_name=model_name,
                image_path=self.test_image,
                text_prompt=self.test_prompt,
                duration=5.0,
                resolution=(512, 512)
            )
            
            # Verify result
            self.assertEqual(result["status"], "success")
            self.assertIn("video_path", result)
            self.assertTrue(Path(result["video_path"]).exists())
            self.assertEqual(result["model"], model_name)
            
            print(f"✅ {model_name} test passed")
            
        except Exception as e:
            self.fail(f"{model_name} test failed: {e}")
    
    def test_luma_ray_flash_2(self):
        """Test luma-ray-flash-2 model."""
        model_name = "luma-ray-flash-2"
        
        try:
            # Check if model is registered
            self.assertIn(model_name, ModelRegistry.list_models())
            
            # Run inference
            result = self.runner.run(
                model_name=model_name,
                image_path=self.test_image,
                text_prompt=self.test_prompt,
                duration=5.0,
                resolution=(512, 512)
            )
            
            # Verify result
            self.assertEqual(result["status"], "success")
            self.assertIn("video_path", result)
            self.assertTrue(Path(result["video_path"]).exists())
            self.assertEqual(result["model"], model_name)
            
            print(f"✅ {model_name} test passed")
            
        except Exception as e:
            self.fail(f"{model_name} test failed: {e}")
    
    def test_luma_ray_2(self):
        """Test luma-ray-2 model."""
        model_name = "luma-ray-2"
        
        try:
            # Check if model is registered
            self.assertIn(model_name, ModelRegistry.list_models())
            
            # Run inference
            result = self.runner.run(
                model_name=model_name,
                image_path=self.test_image,
                text_prompt=self.test_prompt,
                duration=5.0,
                resolution=(512, 512)
            )
            
            # Verify result
            self.assertEqual(result["status"], "success")
            self.assertIn("video_path", result)
            self.assertTrue(Path(result["video_path"]).exists())
            self.assertEqual(result["model"], model_name)
            
            print(f"✅ {model_name} test passed")
            
        except Exception as e:
            self.fail(f"{model_name} test failed: {e}")
    
    def test_model_registry_consistency(self):
        """Test that model registry is consistent."""
        # Get all registered models
        models = ModelRegistry.list_models()
        
        # Should have exactly 3 Luma models
        self.assertEqual(len(models), 3)
        
        # All should be Luma models
        for model_name in models:
            self.assertTrue(model_name.startswith("luma-"))
            
        # All should support text+image input
        for model_name, info in models.items():
            self.assertTrue(info["supports_text_image"])
            self.assertEqual(info["status"], "✅ Compatible")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test outputs."""
        # Remove test output directory
        import shutil
        if Path("./test_outputs").exists():
            shutil.rmtree("./test_outputs")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
