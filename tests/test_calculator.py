#!/usr/bin/env python3
"""
Test Suite for LLM Inference Calculator

Comprehensive tests for all calculator functionality including:
- Basic calculations
- Model specifications
- Hardware compatibility
- Edge cases and error handling
- Performance validation
"""

import unittest
import sys
import os

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    LLMInferenceCalculator,
    DeploymentMode,
    PrecisionMode,
    InferenceResult,
    get_model_specs,
    get_hardware_specs,
    validate_inputs,
    MODEL_DATABASE,
    HARDWARE_DATABASE
)


class TestLLMInferenceCalculator(unittest.TestCase):
    """Test cases for LLMInferenceCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = LLMInferenceCalculator()
    
    def test_basic_calculation(self):
        """Test basic inference calculation."""
        result = self.calculator.calculate_inference(
            model_name="llama-2-7b",
            num_tokens=100,
            batch_size=1,
            hardware_name="rtx-4090",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        
        self.assertIsInstance(result, InferenceResult)
        self.assertGreater(result.latency, 0)
        self.assertGreater(result.memory_usage, 0)
        self.assertGreater(result.cost_per_request, 0)
        self.assertGreater(result.throughput, 0)
        self.assertIsInstance(result.hardware_compatible, bool)
        self.assertIsInstance(result.optimization_suggestions, list)
    
    def test_model_compatibility(self):
        """Test model compatibility with different hardware."""
        # Test compatible configuration
        result_compatible = self.calculator.calculate_inference(
            model_name="llama-2-7b",
            num_tokens=100,
            batch_size=1,
            hardware_name="rtx-4090",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        self.assertTrue(result_compatible.hardware_compatible)
        
        # Test potentially incompatible configuration (large model on small hardware)
        result_incompatible = self.calculator.calculate_inference(
            model_name="llama-2-65b",
            num_tokens=100,
            batch_size=1,
            hardware_name="rtx-3080",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        # Should still return a result but may not be compatible
        self.assertIsInstance(result_incompatible, InferenceResult)
    
    def test_batch_size_scaling(self):
        """Test that batch size affects performance correctly."""
        batch_1 = self.calculator.calculate_inference(
            model_name="llama-2-7b",
            num_tokens=100,
            batch_size=1,
            hardware_name="rtx-4090",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        
        batch_8 = self.calculator.calculate_inference(
            model_name="llama-2-7b",
            num_tokens=100,
            batch_size=8,
            hardware_name="rtx-4090",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        
        # Higher batch size should increase throughput but also latency
        self.assertGreater(batch_8.throughput, batch_1.throughput)
        self.assertGreater(batch_8.latency, batch_1.latency)
        self.assertGreater(batch_8.memory_usage, batch_1.memory_usage)
    
    def test_precision_modes(self):
        """Test different precision modes."""
        fp16_result = self.calculator.calculate_inference(
            model_name="llama-2-7b",
            num_tokens=100,
            batch_size=1,
            hardware_name="rtx-4090",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        
        int8_result = self.calculator.calculate_inference(
            model_name="llama-2-7b",
            num_tokens=100,
            batch_size=1,
            hardware_name="rtx-4090",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.INT8
        )
        
        int4_result = self.calculator.calculate_inference(
            model_name="llama-2-7b",
            num_tokens=100,
            batch_size=1,
            hardware_name="rtx-4090",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.INT4
        )
        
        # Lower precision should use less memory
        self.assertGreater(fp16_result.memory_usage, int8_result.memory_usage)
        self.assertGreater(int8_result.memory_usage, int4_result.memory_usage)
    
    def test_deployment_modes(self):
        """Test different deployment modes."""
        local_result = self.calculator.calculate_inference(
            model_name="llama-2-7b",
            num_tokens=100,
            batch_size=1,
            hardware_name="rtx-4090",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        
        cloud_result = self.calculator.calculate_inference(
            model_name="llama-2-7b",
            num_tokens=100,
            batch_size=1,
            hardware_name="rtx-4090",
            deployment_mode=DeploymentMode.CLOUD,
            precision_mode=PrecisionMode.FP16
        )
        
        # Cloud deployment should have higher latency due to network overhead
        self.assertGreater(cloud_result.latency, local_result.latency)
        # Cloud deployment should have higher cost
        self.assertGreater(cloud_result.cost_per_request, local_result.cost_per_request)
    
    def test_token_scaling(self):
        """Test that token count affects performance correctly."""
        tokens_100 = self.calculator.calculate_inference(
            model_name="llama-2-7b",
            num_tokens=100,
            batch_size=1,
            hardware_name="rtx-4090",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        
        tokens_500 = self.calculator.calculate_inference(
            model_name="llama-2-7b",
            num_tokens=500,
            batch_size=1,
            hardware_name="rtx-4090",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        
        # More tokens should increase latency and cost
        self.assertGreater(tokens_500.latency, tokens_100.latency)
        self.assertGreater(tokens_500.cost_per_request, tokens_100.cost_per_request)
    
    def test_model_comparison(self):
        """Test model comparison functionality."""
        models = ["llama-2-7b", "llama-2-13b"]
        comparison = self.calculator.compare_models(
            models=models,
            num_tokens=100,
            batch_size=1,
            hardware_name="rtx-4090",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        
        self.assertEqual(len(comparison), len(models))
        for model_name, result in comparison.items():
            self.assertIn(model_name, models)
            self.assertIsInstance(result, InferenceResult)
    
    def test_hardware_comparison(self):
        """Test hardware comparison functionality."""
        hardware_list = ["rtx-4090", "a100-40gb"]
        comparison = self.calculator.compare_hardware(
            hardware_list=hardware_list,
            model_name="llama-2-7b",
            num_tokens=100,
            batch_size=1,
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        
        self.assertEqual(len(comparison), len(hardware_list))
        for hw_name, result in comparison.items():
            self.assertIn(hw_name, hardware_list)
            self.assertIsInstance(result, InferenceResult)
    
    def test_optimization_suggestions(self):
        """Test optimization suggestions generation."""
        # Test with a configuration that should generate suggestions
        result = self.calculator.calculate_inference(
            model_name="llama-2-65b",  # Large model
            num_tokens=1000,  # Many tokens
            batch_size=1,  # Small batch
            hardware_name="rtx-3080",  # Limited hardware
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        
        # Should have optimization suggestions
        self.assertGreater(len(result.optimization_suggestions), 0)
        
        # Check for common suggestion types
        suggestions_text = " ".join(result.optimization_suggestions).lower()
        self.assertTrue(
            any(keyword in suggestions_text for keyword in 
                ["quantization", "batch", "hardware", "model", "precision"])
        )


class TestModelSpecs(unittest.TestCase):
    """Test cases for model specifications."""
    
    def test_get_model_specs(self):
        """Test getting model specifications."""
        specs = get_model_specs("llama-2-7b")
        
        self.assertIsNotNone(specs)
        self.assertGreater(specs.parameters, 0)
        self.assertGreater(specs.memory_fp16, 0)
        self.assertGreater(specs.memory_int8, 0)
        self.assertGreater(specs.memory_int4, 0)
        self.assertIsInstance(specs.capabilities, list)
    
    def test_invalid_model(self):
        """Test handling of invalid model names."""
        with self.assertRaises(ValueError):
            get_model_specs("invalid-model")
    
    def test_all_models_in_database(self):
        """Test that all models in database are accessible."""
        for model_name in MODEL_DATABASE.keys():
            specs = get_model_specs(model_name)
            self.assertIsNotNone(specs)
            self.assertGreater(specs.parameters, 0)


class TestHardwareSpecs(unittest.TestCase):
    """Test cases for hardware specifications."""
    
    def test_get_hardware_specs(self):
        """Test getting hardware specifications."""
        specs = get_hardware_specs("rtx-4090")
        
        self.assertIsNotNone(specs)
        self.assertGreater(specs.memory, 0)
        self.assertGreater(specs.compute_capability, 0)
        self.assertGreater(specs.memory_bandwidth, 0)
        self.assertIsInstance(specs.hardware_type.value, str)
    
    def test_invalid_hardware(self):
        """Test handling of invalid hardware names."""
        with self.assertRaises(ValueError):
            get_hardware_specs("invalid-hardware")
    
    def test_all_hardware_in_database(self):
        """Test that all hardware in database is accessible."""
        for hw_name in HARDWARE_DATABASE.keys():
            specs = get_hardware_specs(hw_name)
            self.assertIsNotNone(specs)
            self.assertGreater(specs.memory, 0)


class TestInputValidation(unittest.TestCase):
    """Test cases for input validation."""
    
    def test_valid_inputs(self):
        """Test validation with valid inputs."""
        # Should not raise any exceptions
        validate_inputs(
            model_name="llama-2-7b",
            num_tokens=100,
            batch_size=1,
            hardware_name="rtx-4090"
        )
    
    def test_invalid_model_name(self):
        """Test validation with invalid model name."""
        with self.assertRaises(ValueError):
            validate_inputs(
                model_name="invalid-model",
                num_tokens=100,
                batch_size=1,
                hardware_name="rtx-4090"
            )
    
    def test_invalid_hardware_name(self):
        """Test validation with invalid hardware name."""
        with self.assertRaises(ValueError):
            validate_inputs(
                model_name="llama-2-7b",
                num_tokens=100,
                batch_size=1,
                hardware_name="invalid-hardware"
            )
    
    def test_invalid_token_count(self):
        """Test validation with invalid token count."""
        with self.assertRaises(ValueError):
            validate_inputs(
                model_name="llama-2-7b",
                num_tokens=0,  # Invalid
                batch_size=1,
                hardware_name="rtx-4090"
            )
        
        with self.assertRaises(ValueError):
            validate_inputs(
                model_name="llama-2-7b",
                num_tokens=-10,  # Invalid
                batch_size=1,
                hardware_name="rtx-4090"
            )
    
    def test_invalid_batch_size(self):
        """Test validation with invalid batch size."""
        with self.assertRaises(ValueError):
            validate_inputs(
                model_name="llama-2-7b",
                num_tokens=100,
                batch_size=0,  # Invalid
                hardware_name="rtx-4090"
            )
        
        with self.assertRaises(ValueError):
            validate_inputs(
                model_name="llama-2-7b",
                num_tokens=100,
                batch_size=-5,  # Invalid
                hardware_name="rtx-4090"
            )


class TestEdgeCases(unittest.TestCase):
    """Test cases for edge cases and boundary conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = LLMInferenceCalculator()
    
    def test_very_large_token_count(self):
        """Test with very large token count."""
        result = self.calculator.calculate_inference(
            model_name="llama-2-7b",
            num_tokens=10000,  # Very large
            batch_size=1,
            hardware_name="a100-80gb",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        
        self.assertIsInstance(result, InferenceResult)
        self.assertGreater(result.latency, 0)
    
    def test_very_large_batch_size(self):
        """Test with very large batch size."""
        result = self.calculator.calculate_inference(
            model_name="llama-2-7b",
            num_tokens=100,
            batch_size=64,  # Very large
            hardware_name="a100-80gb",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        
        self.assertIsInstance(result, InferenceResult)
        self.assertGreater(result.memory_usage, 0)
    
    def test_minimum_values(self):
        """Test with minimum valid values."""
        result = self.calculator.calculate_inference(
            model_name="llama-2-7b",
            num_tokens=1,  # Minimum
            batch_size=1,  # Minimum
            hardware_name="rtx-4090",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        
        self.assertIsInstance(result, InferenceResult)
        self.assertGreater(result.latency, 0)
        self.assertGreater(result.cost_per_request, 0)
    
    def test_cpu_only_inference(self):
        """Test CPU-only inference."""
        result = self.calculator.calculate_inference(
            model_name="llama-2-7b",
            num_tokens=100,
            batch_size=1,
            hardware_name="cpu-32gb",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        
        self.assertIsInstance(result, InferenceResult)
        # CPU inference should be much slower
        self.assertGreater(result.latency, 1.0)


class TestPerformanceValidation(unittest.TestCase):
    """Test cases for performance validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = LLMInferenceCalculator()
    
    def test_latency_scaling(self):
        """Test that latency scales appropriately with model size."""
        small_model = self.calculator.calculate_inference(
            model_name="llama-2-7b",
            num_tokens=100,
            batch_size=1,
            hardware_name="rtx-4090",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        
        large_model = self.calculator.calculate_inference(
            model_name="llama-2-13b",
            num_tokens=100,
            batch_size=1,
            hardware_name="rtx-4090",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        
        # Larger model should have higher latency
        self.assertGreater(large_model.latency, small_model.latency)
        # Larger model should use more memory
        self.assertGreater(large_model.memory_usage, small_model.memory_usage)
    
    def test_memory_usage_bounds(self):
        """Test that memory usage is within reasonable bounds."""
        result = self.calculator.calculate_inference(
            model_name="llama-2-7b",
            num_tokens=100,
            batch_size=1,
            hardware_name="rtx-4090",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        
        # Memory usage should be reasonable for 7B model
        self.assertGreater(result.memory_usage, 10)  # At least 10GB
        self.assertLess(result.memory_usage, 30)     # Less than 30GB
    
    def test_cost_reasonableness(self):
        """Test that costs are within reasonable ranges."""
        result = self.calculator.calculate_inference(
            model_name="llama-2-7b",
            num_tokens=100,
            batch_size=1,
            hardware_name="rtx-4090",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        
        # Cost should be reasonable (not too high or too low)
        self.assertGreater(result.cost_per_request, 0.0001)  # At least $0.0001
        self.assertLess(result.cost_per_request, 1.0)        # Less than $1.00


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = LLMInferenceCalculator()
    
    def test_complete_workflow(self):
        """Test a complete analysis workflow."""
        # Step 1: Basic calculation
        result = self.calculator.calculate_inference(
            model_name="llama-2-7b",
            num_tokens=100,
            batch_size=1,
            hardware_name="rtx-4090",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        self.assertIsInstance(result, InferenceResult)
        
        # Step 2: Model comparison
        models = ["llama-2-7b", "llama-2-13b"]
        comparison = self.calculator.compare_models(
            models=models,
            num_tokens=100,
            batch_size=1,
            hardware_name="rtx-4090",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        self.assertEqual(len(comparison), 2)
        
        # Step 3: Hardware comparison
        hardware_list = ["rtx-4090", "a100-40gb"]
        hw_comparison = self.calculator.compare_hardware(
            hardware_list=hardware_list,
            model_name="llama-2-7b",
            num_tokens=100,
            batch_size=1,
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        self.assertEqual(len(hw_comparison), 2)
        
        # Step 4: Optimization
        optimal_batch = self.calculator.optimize_batch_size(
            model_name="llama-2-7b",
            num_tokens=100,
            hardware_name="rtx-4090",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16,
            target_metric="throughput"
        )
        self.assertGreater(optimal_batch, 0)
    
    def test_error_recovery(self):
        """Test error handling and recovery."""
        # Test with invalid inputs but catch exceptions
        try:
            self.calculator.calculate_inference(
                model_name="invalid-model",
                num_tokens=100,
                batch_size=1,
                hardware_name="rtx-4090",
                deployment_mode=DeploymentMode.LOCAL,
                precision_mode=PrecisionMode.FP16
            )
            self.fail("Should have raised ValueError")
        except ValueError:
            pass  # Expected
        
        # Calculator should still work after error
        result = self.calculator.calculate_inference(
            model_name="llama-2-7b",
            num_tokens=100,
            batch_size=1,
            hardware_name="rtx-4090",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        self.assertIsInstance(result, InferenceResult)


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)