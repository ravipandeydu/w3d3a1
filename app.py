#!/usr/bin/env python3
"""
LLM Inference Calculator - Main Application

A comprehensive tool for estimating LLM inference costs, latency, and memory usage.
Supports multiple models, hardware configurations, and deployment modes.

Usage:
    python app.py --model llama-2-7b --tokens 100 --hardware gtx-1650
    python app.py --interactive
    python app.py --scenario chatbot
"""

import argparse
import json
import sys
from typing import Dict, Any, List

from src import (
    LLMInferenceCalculator,
    DeploymentMode,
    PrecisionMode,
    get_model_specs,
    get_hardware_specs,
    format_results,
    validate_inputs,
    MODEL_DATABASE,
    HARDWARE_DATABASE
)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="LLM Inference Calculator - Estimate costs, latency, and memory usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic calculation
  python app.py --model llama-2-7b --tokens 100 --hardware gtx-1650
  
  # With custom parameters
  python app.py --model gpt-4 --tokens 500 --batch-size 4 --hardware a100-80gb --deployment cloud
  
  # Interactive mode
  python app.py --interactive
  
  # Predefined scenarios
  python app.py --scenario chatbot
  python app.py --scenario code-assistant
  python app.py --scenario research
  
  # Compare models
  python app.py --compare --models llama-2-7b llama-2-13b gpt-4 --tokens 100 --hardware gtx-1650
  
  # Hardware recommendations
  python app.py --recommend-hardware --model llama-2-13b --budget 5000
  
  # Export results
  python app.py --model llama-2-7b --tokens 100 --hardware gtx-1650 --export results.json
"""
    )
    
    # Model selection
    parser.add_argument(
        "--model", 
        type=str,
        help="Model to analyze (e.g., llama-2-7b, llama-2-13b, gpt-4)"
    )
    
    # Input parameters
    parser.add_argument(
        "--tokens", 
        type=int, 
        default=100,
        help="Number of tokens to generate (default: 100)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1,
        help="Batch size for inference (default: 1)"
    )
    
    parser.add_argument(
        "--hardware", 
        type=str,
        help="Hardware configuration (e.g., gtx-1650, rtx-4090, a100-80gb, cpu-32gb)"
    )
    
    parser.add_argument(
        "--deployment", 
        type=str, 
        choices=["local", "cloud", "edge", "api"],
        default="local",
        help="Deployment mode (default: local)"
    )
    
    parser.add_argument(
        "--precision", 
        type=str, 
        choices=["fp16", "int8", "int4"],
        default="fp16",
        help="Model precision (default: fp16)"
    )
    
    # Operation modes
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--scenario", 
        type=str,
        choices=["chatbot", "code-assistant", "research"],
        help="Run predefined scenario analysis"
    )
    
    parser.add_argument(
        "--compare", 
        action="store_true",
        help="Compare multiple models"
    )
    
    parser.add_argument(
        "--models", 
        nargs="+",
        help="List of models to compare (use with --compare)"
    )
    
    parser.add_argument(
        "--recommend-hardware", 
        action="store_true",
        help="Get hardware recommendations"
    )
    
    parser.add_argument(
        "--budget", 
        type=float,
        help="Budget constraint for hardware recommendations (USD)"
    )
    
    # Output options
    parser.add_argument(
        "--export", 
        type=str,
        help="Export results to JSON file"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show detailed output"
    )
    
    parser.add_argument(
        "--list-models", 
        action="store_true",
        help="List available models"
    )
    
    parser.add_argument(
        "--list-hardware", 
        action="store_true",
        help="List available hardware configurations"
    )
    
    return parser


def list_models():
    """List all available models."""
    print("\n📋 Available Models:")
    print("=" * 50)
    
    categories = {
        "Small Models (7B)": ["llama-2-7b", "mistral-7b", "code-llama-7b"],
        "Medium Models (13B)": ["llama-2-13b", "vicuna-13b", "code-llama-13b"],
        "Large Models (30B+)": ["llama-2-30b", "llama-2-65b"],
        "API Models": ["gpt-4", "claude-3-opus", "palm-2"]
    }
    
    for category, models in categories.items():
        print(f"\n{category}:")
        for model in models:
            if model in MODEL_DATABASE:
                specs = MODEL_DATABASE[model]
                print(f"  • {model:<20} - {specs.parameters:>8} params, {specs.memory_fp16:>6.1f}GB memory")


def list_hardware():
    """List all available hardware configurations."""
    print("\n🖥️  Available Hardware:")
    print("=" * 50)
    
    categories = {
        "Consumer GPUs": ["gtx-1650", "rtx-3080", "rtx-3090", "rtx-4080", "rtx-4090"],
        "Professional GPUs": ["v100", "a100-40gb", "a100-80gb", "h100"],
        "CPU Configurations": ["cpu-16gb", "cpu-32gb", "cpu-64gb"],
        "Apple Silicon": ["m1-pro", "m1-max", "m2-ultra"]
    }
    
    for category, hardware_list in categories.items():
        print(f"\n{category}:")
        for hw in hardware_list:
            if hw in HARDWARE_DATABASE:
                specs = HARDWARE_DATABASE[hw]
                print(f"  • {hw:<15} - {specs.memory:>6.1f}GB memory, {specs.compute_capability:>6.1f} TFLOPS")


def run_scenario(scenario_name: str, calculator: LLMInferenceCalculator):
    """Run predefined scenario analysis."""
    scenarios = {
        "chatbot": {
            "name": "Customer Service Chatbot",
            "model": "llama-2-7b",
            "tokens": 150,
            "batch_size": 8,
            "hardware": "gtx-1650",
            "deployment": "local",
            "description": "High-volume customer service with fast response times"
        },
        "code-assistant": {
            "name": "Code Generation Assistant",
            "model": "code-llama-13b",
            "tokens": 200,
            "batch_size": 2,
            "hardware": "a100-40gb",
            "deployment": "cloud",
            "description": "IDE plugin for code completion and generation"
        },
        "research": {
            "name": "Research Analysis Platform",
            "model": "gpt-4",
            "tokens": 1000,
            "batch_size": 1,
            "hardware": "api",
            "deployment": "api",
            "description": "High-quality analysis for research and professional use"
        }
    }
    
    if scenario_name not in scenarios:
        print(f"❌ Unknown scenario: {scenario_name}")
        print(f"Available scenarios: {', '.join(scenarios.keys())}")
        return
    
    scenario = scenarios[scenario_name]
    print(f"\n🎯 Scenario: {scenario['name']}")
    print(f"📝 Description: {scenario['description']}")
    print("=" * 60)
    
    # Run calculation
    try:
        if scenario['deployment'] == 'api':
            # For API scenarios, show cost estimation
            print(f"\n📊 API Cost Estimation:")
            print(f"Model: {scenario['model']}")
            print(f"Tokens per request: {scenario['tokens']}")
            
            # Estimate API costs (approximate)
            api_costs = {
                "gpt-4": 0.06,  # per 1K tokens
                "claude-3-opus": 0.075,
                "palm-2": 0.025
            }
            
            cost_per_1k = api_costs.get(scenario['model'], 0.05)
            cost_per_request = (scenario['tokens'] / 1000) * cost_per_1k
            
            print(f"Cost per request: ${cost_per_request:.4f}")
            print(f"Cost per 1K requests: ${cost_per_request * 1000:.2f}")
            print(f"Cost per 1M requests: ${cost_per_request * 1000000:.2f}")
            
        else:
            result = calculator.calculate(
                model_size=scenario['model'],
                tokens=scenario['tokens'],
                batch_size=scenario['batch_size'],
                hardware_type=scenario['hardware'],
                deployment_mode=scenario['deployment'],
                precision_mode="fp16"
            )
            
            print(format_results(result))
            
            # Add scenario-specific insights
            print(f"\n💡 Scenario Insights:")
            if scenario_name == "chatbot":
                print(f"• Estimated concurrent users: {result.tokens_per_second * 10:.0f}")
                print(f"• Daily operating cost: ${result.cost_per_request * 10000:.2f} (10K requests)")
                print(f"• Response time target: < 2 seconds ✓" if result.latency_ms < 2000 else "• Response time target: < 2 seconds ❌")
                
            elif scenario_name == "code-assistant":
                print(f"• Code completions per hour: {3600000 / result.latency_ms:.0f}")
                print(f"• Monthly cost per developer: ${result.cost_per_request * 500:.2f} (500 requests/month)")
                print(f"• IDE integration latency: < 1 second ✓" if result.latency_ms < 1000 else "• IDE integration latency: < 1 second ❌")
                
    except Exception as e:
        print(f"❌ Error running scenario: {e}")


def compare_models(models: List[str], tokens: int, hardware: str, calculator: LLMInferenceCalculator):
    """Compare multiple models."""
    print(f"\n🔍 Model Comparison")
    print(f"Tokens: {tokens}, Hardware: {hardware}")
    print("=" * 80)
    
    results = []
    
    for model in models:
        try:
            result = calculator.calculate(
                model_size=model,
                tokens=tokens,
                batch_size=1,
                hardware_type=hardware,
                deployment_mode="local",
                precision_mode="fp16"
            )
            results.append((model, result))
        except Exception as e:
            print(f"❌ Error with {model}: {e}")
    
    if not results:
        print("No valid results to compare.")
        return
    
    # Print comparison table
    print(f"{'Model':<20} {'Latency':<12} {'Memory':<12} {'Cost/Req':<12} {'Throughput':<12}")
    print("-" * 80)
    
    for model, result in results:
        print(f"{model:<20} {result.latency:<11.2f}s {result.memory_usage:<11.1f}GB "
              f"${result.cost_per_request:<11.4f} {result.throughput:<11.1f}/s")
    
    # Find best options
    fastest = min(results, key=lambda x: x[1].latency)
    cheapest = min(results, key=lambda x: x[1].cost_per_request)
    most_efficient = min(results, key=lambda x: x[1].memory_usage)
    
    print(f"\n🏆 Best Options:")
    print(f"• Fastest: {fastest[0]} ({fastest[1].latency:.2f}s)")
    print(f"• Cheapest: {cheapest[0]} (${cheapest[1].cost_per_request:.4f}/request)")
    print(f"• Most Memory Efficient: {most_efficient[0]} ({most_efficient[1].memory_usage:.1f}GB)")


def recommend_hardware(model_name: str, budget: float, calculator: LLMInferenceCalculator):
    """Recommend hardware based on model and budget."""
    print(f"\n🎯 Hardware Recommendations")
    print(f"Model: {model_name}, Budget: ${budget:,.2f}")
    print("=" * 50)
    
    try:
        model_specs = get_model_specs(model_name)
        print(f"Model Requirements: {model_specs.memory_fp16:.1f}GB memory (FP16)")
        
        # Test different hardware options
        hardware_options = []
        
        for hw_name, hw_specs in HARDWARE_DATABASE.items():
            if hw_specs.price_usd and hw_specs.price_usd <= budget:
                try:
                    result = calculator.calculate(
                        model_size=model_name,
                        tokens=100,
                        batch_size=1,
                        hardware_type=hw_name,
                        deployment_mode="local",
                        precision_mode="fp16"
                    )
                    
                    hardware_options.append({
                        'name': hw_name,
                        'price': hw_specs.price_usd,
                        'performance': result.tokens_per_second,
                        'latency': result.latency_ms / 1000,  # Convert to seconds
                        'memory_usage': result.memory_gb,
                        'compatible': result.memory_gb <= hw_specs.memory
                    })
                except:
                    continue
        
        if not hardware_options:
            print("❌ No compatible hardware found within budget.")
            return
        
        # Sort by performance/price ratio
        compatible_options = [opt for opt in hardware_options if opt['compatible']]
        
        if compatible_options:
            compatible_options.sort(key=lambda x: x['performance'] / x['price'], reverse=True)
            
            print(f"\n✅ Compatible Options (within budget):")
            print(f"{'Hardware':<15} {'Price':<10} {'Performance':<12} {'Latency':<10} {'Value':<10}")
            print("-" * 65)
            
            for opt in compatible_options[:5]:  # Top 5
                value_score = opt['performance'] / opt['price'] * 1000
                print(f"{opt['name']:<15} ${opt['price']:<9,.0f} {opt['performance']:<11.1f}/s "
                      f"{opt['latency']:<9.2f}s {value_score:<9.1f}")
            
            print(f"\n🏆 Recommended: {compatible_options[0]['name']} "
                  f"(${compatible_options[0]['price']:,.0f})")
        else:
            print("❌ No hardware within budget can run this model comfortably.")
            print("\n💡 Consider:")
            print("• Using quantization (INT8/INT4)")
            print("• Increasing budget")
            print("• Choosing a smaller model")
            
    except Exception as e:
        print(f"❌ Error generating recommendations: {e}")


def interactive_mode(calculator: LLMInferenceCalculator):
    """Run calculator in interactive mode."""
    print("\n🚀 LLM Inference Calculator - Interactive Mode")
    print("=" * 50)
    print("Type 'help' for commands, 'quit' to exit")
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            elif command == 'help':
                print("\n📚 Available Commands:")
                print("• calc - Run inference calculation")
                print("• compare - Compare models")
                print("• recommend - Get hardware recommendations")
                print("• models - List available models")
                print("• hardware - List available hardware")
                print("• scenario <name> - Run predefined scenario")
                print("• quit - Exit")
                
            elif command == 'models':
                list_models()
                
            elif command == 'hardware':
                list_hardware()
                
            elif command == 'calc':
                # Get parameters interactively
                model = input("Model name: ").strip()
                tokens = int(input("Number of tokens (default 100): ") or "100")
                batch_size = int(input("Batch size (default 1): ") or "1")
                hardware = input("Hardware: ").strip()
                
                result = calculator.calculate(
                    model_size=model,
                    tokens=tokens,
                    batch_size=batch_size,
                    hardware_type=hardware,
                    deployment_mode="local",
                    precision_mode="fp16"
                )
                
                print(format_results(result))
                
            elif command.startswith('scenario '):
                scenario_name = command.split(' ', 1)[1]
                run_scenario(scenario_name, calculator)
                
            elif command == 'compare':
                models = input("Models to compare (space-separated): ").strip().split()
                tokens = int(input("Number of tokens (default 100): ") or "100")
                hardware = input("Hardware: ").strip()
                compare_models(models, tokens, hardware, calculator)
                
            elif command == 'recommend':
                model = input("Model name: ").strip()
                budget = float(input("Budget (USD): "))
                recommend_hardware(model, budget, calculator)
                
            else:
                print(f"❌ Unknown command: {command}. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main application entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle list commands
    if args.list_models:
        list_models()
        return
        
    if args.list_hardware:
        list_hardware()
        return
    
    # Initialize calculator
    calculator = LLMInferenceCalculator()
    
    # Handle different modes
    if args.interactive:
        interactive_mode(calculator)
        return
    
    if args.scenario:
        run_scenario(args.scenario, calculator)
        return
    
    if args.compare:
        if not args.models:
            print("❌ --models required when using --compare")
            return
        compare_models(args.models, args.tokens, args.hardware, calculator)
        return
    
    if args.recommend_hardware:
        if not args.model:
            print("❌ --model required when using --recommend-hardware")
            return
        if not args.budget:
            print("❌ --budget required when using --recommend-hardware")
            return
        recommend_hardware(args.model, args.budget, calculator)
        return
    
    # Standard calculation mode
    if not args.model or not args.hardware:
        print("❌ --model and --hardware are required for calculation")
        print("Use --help for usage information or --interactive for interactive mode")
        return
    
    try:
        # Run calculation
        result = calculator.calculate(
            model_size=args.model,
            tokens=args.tokens,
            batch_size=args.batch_size,
            hardware_type=args.hardware,
            deployment_mode=args.deployment,
            precision_mode=args.precision
        )
        
        # Display results
        print(format_results(result))
        
        # Export if requested
        if args.export:
            export_data = {
                'inputs': {
                    'model': args.model,
                    'tokens': args.tokens,
                    'batch_size': args.batch_size,
                    'hardware': args.hardware,
                    'deployment': args.deployment,
                    'precision': args.precision
                },
                'results': {
                    'latency_ms': result.latency_ms,
                    'memory_gb': result.memory_gb,
                    'cost_per_request': result.cost_per_request,
                    'tokens_per_second': result.tokens_per_second,
                    'hardware_compatible': result.hardware_compatible
                }
            }
            
            with open(args.export, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"\n💾 Results exported to {args.export}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()