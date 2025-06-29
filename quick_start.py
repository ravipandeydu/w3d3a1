#!/usr/bin/env python3
"""
Quick Start Guide for LLM Inference Calculator

This script provides a guided introduction to the calculator's features.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import (
    LLMInferenceCalculator,
    DeploymentMode,
    PrecisionMode,
    format_results
)


def welcome_message():
    """Display welcome message."""
    print("🚀 Welcome to the LLM Inference Calculator!")
    print("=" * 50)
    print("This tool helps you estimate:")
    print("• Inference latency")
    print("• Memory usage")
    print("• Cost per request")
    print("• Hardware compatibility")
    print("• Optimization suggestions")
    print("\nLet's start with some examples!\n")


def example_1_basic_calculation():
    """Example 1: Basic calculation."""
    print("📊 Example 1: Basic Calculation")
    print("-" * 40)
    print("Calculating inference for Llama-2 7B on GTX 1650...\n")
    
    calculator = LLMInferenceCalculator()
    
    result = calculator.calculate(
        model_size="Llama-2-7B",
        tokens=100,
        batch_size=1,
        hardware_type="GTX_1650",
        deployment_mode="local",
        precision_mode="fp16"
    )
    
    print(format_results(result))
    print("\n" + "=" * 60 + "\n")


def example_2_model_comparison():
    """Example 2: Model comparison."""
    print("🔍 Example 2: Model Comparison")
    print("-" * 40)
    print("Comparing different model sizes...\n")
    
    calculator = LLMInferenceCalculator()
    models = ["Llama-2-7B", "Llama-2-13B", "Mistral-7B"]
    
    print(f"{'Model':<15} {'Latency':<10} {'Memory':<10} {'Cost/Req':<12} {'Throughput':<12}")
    print("-" * 70)
    
    for model in models:
        try:
            result = calculator.calculate(
                model_size=model,
                tokens=100,
                batch_size=1,
                hardware_type="GTX_1650",
                deployment_mode="local",
                precision_mode="fp16"
            )
            
            print(f"{model:<15} {result.latency_ms/1000:<9.2f}s {result.memory_gb:<9.1f}GB "
                  f"${result.cost_per_request:<11.6f} {result.tokens_per_second:<11.1f}/s")
        except Exception as e:
            print(f"{model:<15} Error: {e}")
    
    print("\n💡 Insight: Smaller models are faster and cheaper, but may have lower quality.")
    print("\n" + "=" * 60 + "\n")


def example_3_hardware_comparison():
    """Example 3: Hardware comparison."""
    print("🖥️  Example 3: Hardware Comparison")
    print("-" * 40)
    print("Comparing different hardware options for Llama-2 13B...\n")
    
    calculator = LLMInferenceCalculator()
    hardware_options = ["GTX_1650", "RTX_4090", "A100_40GB", "A100_80GB"]
    
    print(f"{'Hardware':<15} {'Compatible':<12} {'Latency':<10} {'Memory':<10} {'Cost/Req':<12}")
    print("-" * 75)
    
    for hardware in hardware_options:
        try:
            result = calculator.calculate(
                model_size="Llama-2-13B",
                tokens=100,
                batch_size=1,
                hardware_type=hardware,
                deployment_mode="local",
                precision_mode="fp16"
            )
            
            compatible = "✅ Yes" if result.hardware_compatible else "❌ No"
            print(f"{hardware:<15} {compatible:<12} {result.latency_ms/1000:<9.2f}s "
                  f"{result.memory_gb:<9.1f}GB ${result.cost_per_request:<11.6f}")
        except Exception as e:
            print(f"{hardware:<15} Error: {e}")
    
    print("\n💡 Insight: Professional GPUs offer better compatibility for larger models.")
    print("\n" + "=" * 60 + "\n")


def example_4_optimization():
    """Example 4: Optimization suggestions."""
    print("⚡ Example 4: Optimization Suggestions")
    print("-" * 40)
    print("Getting optimization suggestions for a challenging configuration...\n")
    
    calculator = LLMInferenceCalculator()
    
    # Use a configuration that will generate suggestions
    result = calculator.calculate(
        model_size="Llama-2-13B",
        tokens=500,
        batch_size=1,
        hardware_type="RTX_4080",  # Limited memory
        deployment_mode="local",
        precision_mode="fp16"
    )
    
    print(f"Configuration: Llama-2 13B on RTX 4080, 500 tokens")
    print(f"Memory Usage: {result.memory_gb:.1f}GB")
    print(f"Hardware Compatible: {'✅' if result.hardware_compatible else '❌'}")
    print(f"Latency: {result.latency_ms/1000:.2f}s")
    
    # Note: optimization_suggestions is not available in the current InferenceResult
    if not result.hardware_compatible:
        print(f"\n🔧 Optimization Suggestions:")
        print(f"1. Consider using quantization (INT8/INT4)")
        print(f"2. Upgrade to hardware with more memory")
        print(f"3. Reduce batch size")
        print(f"4. Use a smaller model")
    else:
        print("\n✅ Configuration is already optimized!")
    
    print("\n" + "=" * 60 + "\n")


def example_5_cost_analysis():
    """Example 5: Cost analysis."""
    print("💰 Example 5: Cost Analysis")
    print("-" * 40)
    print("Analyzing costs for different usage patterns...\n")
    
    calculator = LLMInferenceCalculator()
    
    result = calculator.calculate(
        model_size="Llama-2-7B",
        tokens=100,
        batch_size=1,
        hardware_type="GTX_1650",
        deployment_mode="local",
        precision_mode="fp16"
    )
    
    usage_patterns = [
        {"name": "Light Usage", "requests_per_day": 100},
        {"name": "Medium Usage", "requests_per_day": 1000},
        {"name": "Heavy Usage", "requests_per_day": 10000},
    ]
    
    print(f"Base cost per request: ${result.cost_per_request:.6f}")
    print(f"\n{'Usage Pattern':<15} {'Daily Cost':<12} {'Monthly Cost':<15} {'Annual Cost':<12}")
    print("-" * 65)
    
    for pattern in usage_patterns:
        daily_cost = result.cost_per_request * pattern["requests_per_day"]
        monthly_cost = daily_cost * 30
        annual_cost = daily_cost * 365
        
        print(f"{pattern['name']:<15} ${daily_cost:<11.2f} ${monthly_cost:<14.2f} ${annual_cost:<11.2f}")
    
    print("\n💡 Insight: On-premise deployment becomes more cost-effective at higher volumes.")
    print("\n" + "=" * 60 + "\n")


def interactive_demo():
    """Interactive demo."""
    print("🎮 Interactive Demo")
    print("-" * 40)
    print("Try your own configuration!\n")
    
    calculator = LLMInferenceCalculator()
    
    try:
        # Get user input
        print("Available models: Llama-2-7B, Llama-2-13B, Mistral-7B, Code-Llama-7B")
        model = input("Enter model name (default: Llama-2-7B): ").strip() or "Llama-2-7B"
        
        tokens = input("Enter number of tokens (default: 100): ").strip()
        tokens = int(tokens) if tokens else 100
        
        print("Available hardware: GTX_1650, RTX_4090, A100_40GB, A100_80GB, CPU_32GB")
        hardware = input("Enter hardware (default: GTX_1650): ").strip() or "GTX_1650"
        
        print("\nCalculating...")
        
        result = calculator.calculate(
            model_size=model,
            tokens=tokens,
            batch_size=1,
            hardware_type=hardware,
            deployment_mode="local",
            precision_mode="fp16"
        )
        
        print("\n" + format_results(result))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your inputs and try again.")
    
    print("\n" + "=" * 60 + "\n")


def next_steps():
    """Show next steps."""
    print("🎯 Next Steps")
    print("-" * 40)
    print("Now that you've seen the basics, here's what you can do next:\n")
    
    print("📚 Explore More:")
    print("• Run 'python app.py --help' for command-line usage")
    print("• Check 'examples/basic_usage.py' for more examples")
    print("• Look at 'examples/advanced_usage.py' for advanced features")
    print("• Read 'scenarios/scenario_analysis.md' for use case studies")
    
    print("\n🔧 Advanced Features:")
    print("• Model comparison: python app.py --compare --models Llama-2-7B Llama-2-13B")
    print("• Hardware recommendations: python app.py --recommend-hardware --model Llama-2-13B")
    print("• Interactive mode: python app.py --interactive")
    print("• Scenario analysis: python app.py --scenario chatbot")
    
    print("\n📖 Documentation:")
    print("• README.md - Project overview and setup")
    print("• research/ - LLM inference research and model comparisons")
    print("• scenarios/ - Use case analysis and recommendations")
    
    print("\n🧪 Testing:")
    print("• Run tests: python run_tests.py")
    print("• Install package: pip install -e .")
    
    print("\n🎉 Happy calculating!")


def main():
    """Main quick start function."""
    try:
        welcome_message()
        example_1_basic_calculation()
        example_2_model_comparison()
        example_3_hardware_comparison()
        example_4_optimization()
        example_5_cost_analysis()
        
        # Ask if user wants interactive demo
        response = input("Would you like to try the interactive demo? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            interactive_demo()
        
        next_steps()
        
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for trying the LLM Inference Calculator!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please make sure all dependencies are installed and try again.")


if __name__ == "__main__":
    main()