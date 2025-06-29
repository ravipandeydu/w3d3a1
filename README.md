# LLM Inference Calculator

A comprehensive calculator that estimates LLM inference costs, latency, and memory usage for different models and deployment scenarios.

## Overview

This project provides tools to estimate the computational requirements and costs for running Large Language Model (LLM) inference across different hardware configurations and deployment modes.

## Features

- **Cost Estimation**: Calculate inference costs per request and token
- **Latency Prediction**: Estimate response times based on model size and hardware
- **Memory Usage**: Calculate VRAM and system memory requirements
- **Hardware Compatibility**: Check if models can run on specific hardware
- **Batch Processing**: Support for batch inference calculations
- **Multiple Deployment Modes**: Cloud, on-premise, and edge deployment scenarios

## Project Structure

```
├── README.md                 # This file
├── research/
│   ├── llm_inference_basics.md   # LLM inference fundamentals
│   └── model_comparison.md       # Comparison of 7B, 13B, and GPT-4 models
├── src/
│   ├── calculator.py             # Main calculator implementation
│   ├── models.py                 # Model specifications and parameters
│   ├── hardware.py               # Hardware configurations
│   └── utils.py                  # Utility functions
├── scenarios/
│   ├── scenario_analysis.md      # Analysis of 3 use cases
│   └── recommendations.md        # Hardware and deployment recommendations
├── examples/
│   ├── basic_usage.py            # Basic calculator usage examples
│   └── advanced_scenarios.py     # Advanced use case examples
├── tests/
│   └── test_calculator.py        # Unit tests
└── requirements.txt              # Python dependencies
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd w3d3a1

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.calculator import LLMInferenceCalculator

# Initialize calculator
calc = LLMInferenceCalculator()

# Calculate inference metrics
result = calc.calculate(
    model_size="7B",
    tokens=1000,
    batch_size=1,
    hardware_type="GTX_1650",
    deployment_mode="local"
)

print(f"Latency: {result.latency_ms}ms")
print(f"Memory: {result.memory_gb}GB")
print(f"Cost: ${result.cost_per_request}")
```

## Supported Models

- **7B Models**: Llama 2 7B, Mistral 7B
- **13B Models**: Llama 2 13B, Vicuna 13B
- **Large Models**: GPT-4, Claude, PaLM

## Supported Hardware

- **Consumer GPUs**: GTX 1650, RTX 3080, RTX 3090, RTX 4080, RTX 4090
- **Professional GPUs**: A100, H100, V100
- **Cloud Instances**: AWS, GCP, Azure GPU instances
- **CPU-only**: Various CPU configurations

## Deployment Modes

- **Local**: On-premise deployment
- **Cloud**: Cloud provider instances
- **Edge**: Mobile and embedded devices
- **Serverless**: Function-as-a-Service platforms

## Documentation

- [LLM Inference Basics](research/llm_inference_basics.md)
- [Model Comparison](research/model_comparison.md)
- [Scenario Analysis](scenarios/scenario_analysis.md)
- [Recommendations](scenarios/recommendations.md)

## Examples

See the `examples/` directory for detailed usage examples and common scenarios.

## Testing

```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.