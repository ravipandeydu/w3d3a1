#!/usr/bin/env python3
"""
Basic Usage Examples for LLM Inference Calculator

This script demonstrates basic usage patterns of the LLM Inference Calculator.
"""

import sys
import os

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    LLMInferenceCalculator,
    DeploymentMode,
    PrecisionMode,
    format_results
)


def example_basic_calculation():
    """Example 1: Basic inference calculation."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Inference Calculation")
    print("=" * 60)
    
    calculator = LLMInferenceCalculator()
    
    # Calculate inference for Llama-2 7B on GTX 1650
    result = calculator.calculate_inference(
        model_name="llama-2-7b",
        num_tokens=100,
        batch_size=1,
        hardware_name="gtx-1650",
        deployment_mode=DeploymentMode.LOCAL,
        precision_mode=PrecisionMode.FP16
    )
    
    print(format_results(result, verbose=True))


def example_batch_processing():
    """Example 2: Batch processing optimization."""
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing Optimization")
    print("=" * 60)
    
    calculator = LLMInferenceCalculator()
    
    # Compare different batch sizes
    batch_sizes = [1, 4, 8, 16]
    
    print(f"{'Batch Size':<12} {'Latency':<12} {'Throughput':<12} {'Cost/Token':<12}")
    print("-" * 50)
    
    for batch_size in batch_sizes:
        result = calculator.calculate_inference(
            model_name="llama-2-7b",
            num_tokens=100,
            batch_size=batch_size,
            hardware_name="gtx-1650",
            deployment_mode=DeploymentMode.LOCAL,
            precision_mode=PrecisionMode.FP16
        )
        
        cost_per_token = result.cost_per_request / 100
        print(f"{batch_size:<12} {result.latency:<11.2f}s {result.throughput:<11.1f}/s ${cost_per_token:<11.6f}")


def example_model_comparison():
    """Example 3: Compare different models."""
    print("\n" + "=" * 60)
    print("Example 3: Model Comparison")
    print("=" * 60)
    
    calculator = LLMInferenceCalculator()
    
    models = ["llama-2-7b", "llama-2-13b", "mistral-7b"]
    
    print(f"{'Model':<15} {'Memory':<10} {'Latency':<10} {'Throughput':<12} {'Cost/Req':<10}")
    print("-" * 70)
    
    for model in models:
        try:
            result = calculator.calculate_inference(
                model_name=model,
                num_tokens=100,
                batch_size=1,
                hardware_name="gtx-1650",
                deployment_mode=DeploymentMode.LOCAL,
                precision_mode=PrecisionMode.FP16
            )
            
            print(f"{model:<15} {result.memory_usage:<9.1f}GB {result.latency:<9.2f}s "
                  f"{result.throughput:<11.1f}/s ${result.cost_per_request:<9.4f}")
        except Exception as e:
            print(f"{model:<15} Error: {e}")


def example_hardware_comparison():
    """Example 4: Compare different hardware."""
    print("\n" + "=" * 60)
    print("Example 4: Hardware Comparison")
    print("=" * 60)
    
    calculator = LLMInferenceCalculator()
    
    hardware_options = ["gtx-1650", "rtx-4090", "a100-40gb", "a100-80gb"]
    
    print(f"{'Hardware':<15} {'Compatible':<12} {'Latency':<10} {'Throughput':<12} {'Cost/Req':<10}")
    print("-" * 75)
    
    for hardware in hardware_options:
        try:
            result = calculator.calculate_inference(
                model_name="llama-2-13b",
                num_tokens=100,
                batch_size=1,
                hardware_name=hardware,
                deployment_mode=DeploymentMode.LOCAL,
                precision_mode=PrecisionMode.FP16
            )
            
            compatible = "✓" if result.hardware_compatible else "❌"
            print(f"{hardware:<15} {compatible:<12} {result.latency:<9.2f}s "
                  f"{result.throughput:<11.1f}/s ${result.cost_per_request:<9.4f}")
        except Exception as e:
            print(f"{hardware:<15} Error: {e}")


def example_precision_modes():
    """Example 5: Compare different precision modes."""
    print("\n" + "=" * 60)
    print("Example 5: Precision Mode Comparison")
    print("=" * 60)
    
    calculator = LLMInferenceCalculator()
    
    precisions = [PrecisionMode.FP16, PrecisionMode.INT8, PrecisionMode.INT4]
    
    print(f"{'Precision':<12} {'Memory':<10} {'Latency':<10} {'Quality':<10} {'Compatible':<12}")
    print("-" * 70)
    
    for precision in precisions:
        try:
            result = calculator.calculate_inference(
                model_name="llama-2-13b",
                num_tokens=100,
                batch_size=1,
                hardware_name="gtx-1650",
                deployment_mode=DeploymentMode.LOCAL,
                precision_mode=precision
            )
            
            # Estimate quality impact
            quality_map = {
                PrecisionMode.FP16: "100%",
                PrecisionMode.INT8: "~98%",
                PrecisionMode.INT4: "~95%"
            }
            
            compatible = "✓" if result.hardware_compatible else "❌"
            print(f"{precision.value:<12} {result.memory_usage:<9.1f}GB {result.latency:<9.2f}s "
                  f"{quality_map[precision]:<10} {compatible:<12}")
        except Exception as e:
            print(f"{precision.value:<12} Error: {e}")


def example_deployment_modes():
    """Example 6: Compare deployment modes."""
    print("\n" + "=" * 60)
    print("Example 6: Deployment Mode Comparison")
    print("=" * 60)
    
    calculator = LLMInferenceCalculator()
    
    deployments = [DeploymentMode.LOCAL, DeploymentMode.CLOUD, DeploymentMode.EDGE]
    
    print(f"{'Deployment':<12} {'Latency':<10} {'Cost/Req':<12} {'Scalability':<12}")
    print("-" * 60)
    
    for deployment in deployments:
        try:
            result = calculator.calculate_inference(
                model_name="llama-2-7b",
                num_tokens=100,
                batch_size=1,
                hardware_name="gtx-1650",
                deployment_mode=deployment,
                precision_mode=PrecisionMode.FP16
            )
            
            # Estimate scalability
            scalability_map = {
                DeploymentMode.LOCAL: "Limited",
                DeploymentMode.CLOUD: "High",
                DeploymentMode.EDGE: "Medium"
            }
            
            print(f"{deployment.value:<12} {result.latency:<9.2f}s "
                  f"${result.cost_per_request:<11.4f} {scalability_map[deployment]:<12}")
        except Exception as e:
            print(f"{deployment.value:<12} Error: {e}")


def example_cost_analysis():
    """Example 7: Detailed cost analysis."""
    print("\n" + "=" * 60)
    print("Example 7: Cost Analysis")
    print("=" * 60)
    
    calculator = LLMInferenceCalculator()
    
    # Calculate costs for different usage patterns
    usage_patterns = [
        {"name": "Light Usage", "requests_per_day": 100, "tokens_per_request": 50},
        {"name": "Medium Usage", "requests_per_day": 1000, "tokens_per_request": 100},
        {"name": "Heavy Usage", "requests_per_day": 10000, "tokens_per_request": 200},
    ]
    
    result = calculator.calculate_inference(
        model_name="llama-2-7b",
        num_tokens=100,  # Base calculation
        batch_size=1,
        hardware_name="gtx-1650",
        deployment_mode=DeploymentMode.LOCAL,
        precision_mode=PrecisionMode.FP16
    )
    
    print(f"Base cost per request: ${result.cost_per_request:.6f}")
    print(f"\n{'Usage Pattern':<15} {'Daily Cost':<12} {'Monthly Cost':<15} {'Annual Cost':<12}")
    print("-" * 70)
    
    for pattern in usage_patterns:
        # Adjust cost based on token count
        token_ratio = pattern["tokens_per_request"] / 100
        adjusted_cost = result.cost_per_request * token_ratio
        
        daily_cost = adjusted_cost * pattern["requests_per_day"]
        monthly_cost = daily_cost * 30
        annual_cost = daily_cost * 365
        
        print(f"{pattern['name']:<15} ${daily_cost:<11.2f} ${monthly_cost:<14.2f} ${annual_cost:<11.2f}")


def example_optimization_suggestions():
    """Example 8: Get optimization suggestions."""
    print("\n" + "=" * 60)
    print("Example 8: Optimization Suggestions")
    print("=" * 60)
    
    calculator = LLMInferenceCalculator()
    
    # Test a configuration that might need optimization
    result = calculator.calculate_inference(
        model_name="llama-2-13b",
        num_tokens=500,
        batch_size=1,
        hardware_name="rtx-4080",  # Limited memory
        deployment_mode=DeploymentMode.LOCAL,
        precision_mode=PrecisionMode.FP16
    )
    
    print(f"Configuration: Llama-2 13B on RTX 4080")
    print(f"Memory Usage: {result.memory_usage:.1f}GB")
    print(f"Hardware Compatible: {'✓' if result.hardware_compatible else '❌'}")
    print(f"\nOptimization Suggestions:")
    
    for i, suggestion in enumerate(result.optimization_suggestions, 1):
        print(f"{i}. {suggestion}")


def main():
    """Run all examples."""
    print("🚀 LLM Inference Calculator - Usage Examples")
    print("=" * 60)
    
    examples = [
        example_basic_calculation,
        example_batch_processing,
        example_model_comparison,
        example_hardware_comparison,
        example_precision_modes,
        example_deployment_modes,
        example_cost_analysis,
        example_optimization_suggestions
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"❌ Error in {example_func.__name__}: {e}")
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()