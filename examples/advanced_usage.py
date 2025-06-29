#!/usr/bin/env python3
"""
Advanced Usage Examples for LLM Inference Calculator

This script demonstrates advanced usage patterns including:
- Multi-model deployments
- Cost optimization strategies
- Performance benchmarking
- Scaling analysis
- ROI calculations
"""

import sys
import os
import json
from typing import List, Dict, Any

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    LLMInferenceCalculator,
    DeploymentMode,
    PrecisionMode,
    format_results,
    get_model_specs,
    get_hardware_specs,
    MODEL_DATABASE,
    HARDWARE_DATABASE
)


def example_multi_model_deployment():
    """Example 1: Multi-model deployment strategy."""
    print("\n" + "=" * 70)
    print("Example 1: Multi-Model Deployment Strategy")
    print("=" * 70)
    
    calculator = LLMInferenceCalculator()
    
    # Define a multi-tier deployment
    deployment_tiers = [
        {
            "name": "Tier 1: High Volume, Simple Tasks",
            "model": "mistral-7b",
            "hardware": "gtx-1650",
            "requests_per_day": 10000,
            "avg_tokens": 50
        },
        {
            "name": "Tier 2: Medium Volume, Complex Tasks",
            "model": "llama-2-13b",
            "hardware": "a100-40gb",
            "requests_per_day": 1000,
            "avg_tokens": 200
        },
        {
            "name": "Tier 3: Low Volume, Premium Tasks",
            "model": "gpt-4",
            "hardware": "api",
            "requests_per_day": 100,
            "avg_tokens": 500
        }
    ]
    
    total_daily_cost = 0
    
    for tier in deployment_tiers:
        print(f"\n{tier['name']}:")
        print(f"Model: {tier['model']}, Hardware: {tier['hardware']}")
        
        if tier['model'] == 'gpt-4' and tier['hardware'] == 'api':
            # API pricing estimation
            cost_per_1k_tokens = 0.06  # GPT-4 pricing
            cost_per_request = (tier['avg_tokens'] / 1000) * cost_per_1k_tokens
            daily_cost = cost_per_request * tier['requests_per_day']
            
            print(f"Cost per request: ${cost_per_request:.4f}")
            print(f"Daily cost: ${daily_cost:.2f}")
            
        else:
            result = calculator.calculate_inference(
                model_name=tier['model'],
                num_tokens=tier['avg_tokens'],
                batch_size=1,
                hardware_name=tier['hardware'],
                deployment_mode=DeploymentMode.LOCAL,
                precision_mode=PrecisionMode.FP16
            )
            
            daily_cost = result.cost_per_request * tier['requests_per_day']
            
            print(f"Latency: {result.latency:.2f}s")
            print(f"Cost per request: ${result.cost_per_request:.6f}")
            print(f"Daily cost: ${daily_cost:.2f}")
        
        total_daily_cost += daily_cost
    
    print(f"\n💰 Total Daily Cost: ${total_daily_cost:.2f}")
    print(f"💰 Total Monthly Cost: ${total_daily_cost * 30:.2f}")
    print(f"💰 Total Annual Cost: ${total_daily_cost * 365:.2f}")


def example_cost_optimization():
    """Example 2: Cost optimization analysis."""
    print("\n" + "=" * 70)
    print("Example 2: Cost Optimization Analysis")
    print("=" * 70)
    
    calculator = LLMInferenceCalculator()
    
    # Compare different optimization strategies
    base_config = {
        "model": "llama-2-13b",
        "tokens": 200,
        "batch_size": 1,
        "hardware": "a100-40gb",
        "deployment": DeploymentMode.LOCAL,
        "precision": PrecisionMode.FP16
    }
    
    optimizations = [
        {"name": "Baseline", "changes": {}},
        {"name": "Quantization (INT8)", "changes": {"precision": PrecisionMode.INT8}},
        {"name": "Quantization (INT4)", "changes": {"precision": PrecisionMode.INT4}},
        {"name": "Batch Processing", "changes": {"batch_size": 8}},
        {"name": "Smaller Model", "changes": {"model": "llama-2-7b"}},
        {"name": "Cheaper Hardware", "changes": {"hardware": "gtx-1650"}},
        {"name": "Combined Optimizations", "changes": {
            "precision": PrecisionMode.INT8,
            "batch_size": 4,
            "model": "llama-2-7b"
        }}
    ]
    
    print(f"{'Strategy':<25} {'Cost/Req':<12} {'Latency':<10} {'Memory':<10} {'Savings':<10}")
    print("-" * 80)
    
    baseline_cost = None
    
    for opt in optimizations:
        config = base_config.copy()
        config.update(opt["changes"])
        
        try:
            result = calculator.calculate_inference(
                model_name=config["model"],
                num_tokens=config["tokens"],
                batch_size=config["batch_size"],
                hardware_name=config["hardware"],
                deployment_mode=config["deployment"],
                precision_mode=config["precision"]
            )
            
            if baseline_cost is None:
                baseline_cost = result.cost_per_request
                savings = "0%"
            else:
                savings_pct = (1 - result.cost_per_request / baseline_cost) * 100
                savings = f"{savings_pct:+.1f}%"
            
            print(f"{opt['name']:<25} ${result.cost_per_request:<11.6f} "
                  f"{result.latency:<9.2f}s {result.memory_usage:<9.1f}GB {savings:<10}")
            
        except Exception as e:
            print(f"{opt['name']:<25} Error: {e}")


def example_performance_benchmarking():
    """Example 3: Performance benchmarking across configurations."""
    print("\n" + "=" * 70)
    print("Example 3: Performance Benchmarking")
    print("=" * 70)
    
    calculator = LLMInferenceCalculator()
    
    # Test different token lengths
    token_lengths = [50, 100, 200, 500, 1000]
    models = ["llama-2-7b", "llama-2-13b"]
    
    for model in models:
        print(f"\n📊 Model: {model}")
        print(f"{'Tokens':<8} {'Latency':<10} {'Throughput':<12} {'Memory':<10} {'Cost/Token':<12}")
        print("-" * 65)
        
        for tokens in token_lengths:
            try:
                result = calculator.calculate_inference(
                    model_name=model,
                    num_tokens=tokens,
                    batch_size=1,
                    hardware_name="gtx-1650",
                    deployment_mode=DeploymentMode.LOCAL,
                    precision_mode=PrecisionMode.FP16
                )
                
                cost_per_token = result.cost_per_request / tokens
                
                print(f"{tokens:<8} {result.latency:<9.2f}s {result.throughput:<11.1f}/s "
                      f"{result.memory_usage:<9.1f}GB ${cost_per_token:<11.6f}")
                
            except Exception as e:
                print(f"{tokens:<8} Error: {e}")


def example_scaling_analysis():
    """Example 4: Scaling analysis for different load patterns."""
    print("\n" + "=" * 70)
    print("Example 4: Scaling Analysis")
    print("=" * 70)
    
    calculator = LLMInferenceCalculator()
    
    # Analyze scaling requirements
    load_scenarios = [
        {"name": "Startup", "peak_rps": 10, "avg_rps": 2},
        {"name": "Growth", "peak_rps": 100, "avg_rps": 20},
        {"name": "Scale", "peak_rps": 1000, "avg_rps": 200},
        {"name": "Enterprise", "peak_rps": 5000, "avg_rps": 1000}
    ]
    
    # Calculate single instance capacity
    result = calculator.calculate_inference(
        model_name="llama-2-7b",
        num_tokens=100,
        batch_size=8,  # Optimized batch size
        hardware_name="gtx-1650",
        deployment_mode=DeploymentMode.LOCAL,
        precision_mode=PrecisionMode.FP16
    )
    
    single_instance_rps = result.throughput / 100  # tokens/s to requests/s
    
    print(f"Single Instance Capacity: {single_instance_rps:.1f} requests/second")
    print(f"\n{'Scenario':<12} {'Peak RPS':<10} {'Instances':<12} {'Monthly Cost':<15} {'Utilization':<12}")
    print("-" * 75)
    
    for scenario in load_scenarios:
        instances_needed = max(1, int(scenario["peak_rps"] / single_instance_rps) + 1)
        monthly_cost = instances_needed * result.cost_per_request * scenario["avg_rps"] * 30 * 24 * 3600
        avg_utilization = (scenario["avg_rps"] / (instances_needed * single_instance_rps)) * 100
        
        print(f"{scenario['name']:<12} {scenario['peak_rps']:<10} {instances_needed:<12} "
              f"${monthly_cost:<14,.0f} {avg_utilization:<11.1f}%")


def example_roi_calculation():
    """Example 5: ROI calculation for different deployment options."""
    print("\n" + "=" * 70)
    print("Example 5: ROI Calculation")
    print("=" * 70)
    
    calculator = LLMInferenceCalculator()
    
    # Compare on-premise vs cloud vs API
    monthly_requests = 100000
    tokens_per_request = 150
    
    deployment_options = [
        {
            "name": "On-Premise (GTX 1650)",
            "model": "llama-2-7b",
            "hardware": "gtx-1650",
            "deployment": DeploymentMode.LOCAL,
            "initial_cost": 2000,  # Hardware cost
            "monthly_fixed": 200   # Power, maintenance
        },
        {
            "name": "Cloud (A100)",
            "model": "llama-2-7b",
            "hardware": "a100-40gb",
            "deployment": DeploymentMode.CLOUD,
            "initial_cost": 0,
            "monthly_fixed": 0
        },
        {
            "name": "API (GPT-4)",
            "model": "gpt-4",
            "hardware": "api",
            "deployment": DeploymentMode.API,
            "initial_cost": 0,
            "monthly_fixed": 0,
            "api_cost_per_1k": 0.06
        }
    ]
    
    print(f"Monthly Volume: {monthly_requests:,} requests, {tokens_per_request} tokens each")
    print(f"\n{'Option':<25} {'Initial':<10} {'Monthly':<12} {'Year 1':<10} {'Year 2':<10} {'Break-even':<12}")
    print("-" * 85)
    
    for option in deployment_options:
        if option["name"].startswith("API"):
            # API pricing
            monthly_variable = (monthly_requests * tokens_per_request / 1000) * option["api_cost_per_1k"]
        else:
            # Calculate using our calculator
            result = calculator.calculate_inference(
                model_name=option["model"],
                num_tokens=tokens_per_request,
                batch_size=1,
                hardware_name=option["hardware"],
                deployment_mode=option["deployment"],
                precision_mode=PrecisionMode.FP16
            )
            monthly_variable = result.cost_per_request * monthly_requests
        
        monthly_total = monthly_variable + option["monthly_fixed"]
        year1_total = option["initial_cost"] + (monthly_total * 12)
        year2_total = year1_total + (monthly_total * 12)
        
        # Calculate break-even vs API option (if not API)
        if not option["name"].startswith("API"):
            api_monthly = (monthly_requests * tokens_per_request / 1000) * 0.06
            if monthly_total < api_monthly:
                breakeven_months = option["initial_cost"] / (api_monthly - monthly_total)
                breakeven = f"{breakeven_months:.1f} months"
            else:
                breakeven = "Never"
        else:
            breakeven = "N/A"
        
        print(f"{option['name']:<25} ${option['initial_cost']:<9,.0f} "
              f"${monthly_total:<11,.0f} ${year1_total:<9,.0f} ${year2_total:<9,.0f} {breakeven:<12}")


def example_hardware_recommendation_engine():
    """Example 6: Advanced hardware recommendation engine."""
    print("\n" + "=" * 70)
    print("Example 6: Hardware Recommendation Engine")
    print("=" * 70)
    
    calculator = LLMInferenceCalculator()
    
    # Define requirements
    requirements = {
        "model": "llama-2-13b",
        "max_latency": 2.0,  # seconds
        "min_throughput": 50,  # tokens/s
        "budget": 10000,  # USD
        "tokens_per_request": 200
    }
    
    print(f"Requirements:")
    print(f"• Model: {requirements['model']}")
    print(f"• Max Latency: {requirements['max_latency']}s")
    print(f"• Min Throughput: {requirements['min_throughput']} tokens/s")
    print(f"• Budget: ${requirements['budget']:,}")
    
    # Test all hardware options
    recommendations = []
    
    for hw_name, hw_specs in HARDWARE_DATABASE.items():
        if hw_specs.price_usd and hw_specs.price_usd <= requirements['budget']:
            try:
                result = calculator.calculate_inference(
                    model_name=requirements['model'],
                    num_tokens=requirements['tokens_per_request'],
                    batch_size=1,
                    hardware_name=hw_name,
                    deployment_mode=DeploymentMode.LOCAL,
                    precision_mode=PrecisionMode.FP16
                )
                
                # Check if requirements are met
                meets_latency = result.latency <= requirements['max_latency']
                meets_throughput = result.throughput >= requirements['min_throughput']
                is_compatible = result.hardware_compatible
                
                if meets_latency and meets_throughput and is_compatible:
                    score = (result.throughput / requirements['min_throughput']) / (hw_specs.price_usd / 1000)
                    
                    recommendations.append({
                        'hardware': hw_name,
                        'price': hw_specs.price_usd,
                        'latency': result.latency,
                        'throughput': result.throughput,
                        'memory_usage': result.memory_usage,
                        'score': score
                    })
                    
            except Exception:
                continue
    
    if recommendations:
        # Sort by score (performance per dollar)
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n✅ Compatible Hardware Options:")
        print(f"{'Hardware':<15} {'Price':<10} {'Latency':<10} {'Throughput':<12} {'Memory':<10} {'Score':<8}")
        print("-" * 80)
        
        for rec in recommendations[:5]:  # Top 5
            print(f"{rec['hardware']:<15} ${rec['price']:<9,.0f} {rec['latency']:<9.2f}s "
                  f"{rec['throughput']:<11.1f}/s {rec['memory_usage']:<9.1f}GB {rec['score']:<7.2f}")
        
        print(f"\n🏆 Recommended: {recommendations[0]['hardware']} (${recommendations[0]['price']:,.0f})")
    else:
        print("\n❌ No hardware meets all requirements within budget.")


def example_batch_optimization():
    """Example 7: Batch size optimization."""
    print("\n" + "=" * 70)
    print("Example 7: Batch Size Optimization")
    print("=" * 70)
    
    calculator = LLMInferenceCalculator()
    
    # Find optimal batch size for different scenarios
    scenarios = [
        {"name": "Low Latency", "priority": "latency", "max_latency": 1.0},
        {"name": "High Throughput", "priority": "throughput", "min_throughput": 200},
        {"name": "Cost Efficient", "priority": "cost", "max_cost": 0.001}
    ]
    
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print(f"{'Batch Size':<12} {'Latency':<10} {'Throughput':<12} {'Cost/Req':<12} {'Status':<10}")
        print("-" * 70)
        
        best_batch = None
        best_score = float('-inf') if scenario['priority'] == 'throughput' else float('inf')
        
        for batch_size in batch_sizes:
            try:
                result = calculator.calculate_inference(
                    model_name="llama-2-7b",
                    num_tokens=100,
                    batch_size=batch_size,
                    hardware_name="gtx-1650",
                    deployment_mode=DeploymentMode.LOCAL,
                    precision_mode=PrecisionMode.FP16
                )
                
                # Check constraints
                status = "✓"
                if scenario['priority'] == 'latency' and result.latency > scenario['max_latency']:
                    status = "❌ Latency"
                elif scenario['priority'] == 'throughput' and result.throughput < scenario['min_throughput']:
                    status = "❌ Throughput"
                elif scenario['priority'] == 'cost' and result.cost_per_request > scenario['max_cost']:
                    status = "❌ Cost"
                
                # Update best option
                if status == "✓":
                    if scenario['priority'] == 'latency' and result.latency < best_score:
                        best_score = result.latency
                        best_batch = batch_size
                    elif scenario['priority'] == 'throughput' and result.throughput > best_score:
                        best_score = result.throughput
                        best_batch = batch_size
                    elif scenario['priority'] == 'cost' and result.cost_per_request < best_score:
                        best_score = result.cost_per_request
                        best_batch = batch_size
                
                print(f"{batch_size:<12} {result.latency:<9.2f}s {result.throughput:<11.1f}/s "
                      f"${result.cost_per_request:<11.6f} {status:<10}")
                
            except Exception as e:
                print(f"{batch_size:<12} Error: {e}")
        
        if best_batch:
            print(f"\n🎯 Optimal batch size: {best_batch}")
        else:
            print(f"\n❌ No batch size meets requirements")


def example_export_analysis():
    """Example 8: Export detailed analysis to JSON."""
    print("\n" + "=" * 70)
    print("Example 8: Export Analysis")
    print("=" * 70)
    
    calculator = LLMInferenceCalculator()
    
    # Comprehensive analysis
    analysis = {
        "timestamp": "2024-01-01T00:00:00Z",
        "analysis_type": "comprehensive_comparison",
        "models": [],
        "summary": {}
    }
    
    models_to_analyze = ["llama-2-7b", "llama-2-13b", "mistral-7b"]
    hardware_to_test = ["gtx-1650", "rtx-4090", "a100-40gb"]
    
    for model in models_to_analyze:
        model_data = {
            "name": model,
            "specifications": get_model_specs(model).__dict__,
            "hardware_results": []
        }
        
        for hardware in hardware_to_test:
            try:
                result = calculator.calculate_inference(
                    model_name=model,
                    num_tokens=100,
                    batch_size=1,
                    hardware_name=hardware,
                    deployment_mode=DeploymentMode.LOCAL,
                    precision_mode=PrecisionMode.FP16
                )
                
                hardware_result = {
                    "hardware": hardware,
                    "hardware_specs": get_hardware_specs(hardware).__dict__,
                    "performance": {
                        "latency": result.latency,
                        "throughput": result.throughput,
                        "memory_usage": result.memory_usage,
                        "cost_per_request": result.cost_per_request,
                        "hardware_compatible": result.hardware_compatible
                    },
                    "optimization_suggestions": result.optimization_suggestions
                }
                
                model_data["hardware_results"].append(hardware_result)
                
            except Exception as e:
                print(f"Error testing {model} on {hardware}: {e}")
        
        analysis["models"].append(model_data)
    
    # Generate summary
    analysis["summary"] = {
        "total_models_tested": len(models_to_analyze),
        "total_hardware_tested": len(hardware_to_test),
        "recommendations": {
            "best_value": "llama-2-7b on GTX 1650",
            "best_performance": "llama-2-13b on A100 40GB",
            "most_cost_effective": "mistral-7b on GTX 1650"
        }
    }
    
    # Export to file
    output_file = "analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"✅ Analysis exported to {output_file}")
    print(f"📊 Analyzed {len(models_to_analyze)} models on {len(hardware_to_test)} hardware configurations")
    print(f"💾 File size: {os.path.getsize(output_file)} bytes")


def main():
    """Run all advanced examples."""
    print("🚀 LLM Inference Calculator - Advanced Usage Examples")
    print("=" * 70)
    
    examples = [
        example_multi_model_deployment,
        example_cost_optimization,
        example_performance_benchmarking,
        example_scaling_analysis,
        example_roi_calculation,
        example_hardware_recommendation_engine,
        example_batch_optimization,
        example_export_analysis
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"❌ Error in {example_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ All advanced examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()