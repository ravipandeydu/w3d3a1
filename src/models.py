#!/usr/bin/env python3
"""
Model Specifications

Defines specifications and parameters for different LLM models.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelSpecs:
    """Specifications for an LLM model."""
    name: str
    parameters: int  # Number of parameters
    layers: int
    hidden_dim: int
    attention_heads: int
    vocab_size: int
    context_length: int
    architecture: str
    
    # Memory requirements (GB)
    fp32_size: float
    fp16_size: float
    int8_size: float
    int4_size: float
    
    # Performance characteristics
    relative_speed: float  # Relative to 7B baseline
    quality_score: float   # Relative quality (0-100)
    
    # Specialized capabilities
    code_optimized: bool = False
    multimodal: bool = False
    instruction_tuned: bool = True


# Model database
MODEL_DATABASE: Dict[str, ModelSpecs] = {
    # 7B Parameter Models
    "7B": ModelSpecs(
        name="Generic 7B",
        parameters=7_000_000_000,
        layers=32,
        hidden_dim=4096,
        attention_heads=32,
        vocab_size=32000,
        context_length=4096,
        architecture="Transformer",
        fp32_size=28.0,
        fp16_size=14.0,
        int8_size=7.0,
        int4_size=3.5,
        relative_speed=1.0,
        quality_score=65.0
    ),
    
    "Llama-2-7B": ModelSpecs(
        name="Llama 2 7B",
        parameters=6_738_415_616,
        layers=32,
        hidden_dim=4096,
        attention_heads=32,
        vocab_size=32000,
        context_length=4096,
        architecture="Llama",
        fp32_size=26.9,
        fp16_size=13.5,
        int8_size=6.7,
        int4_size=3.4,
        relative_speed=1.0,
        quality_score=68.0
    ),
    
    "Mistral-7B": ModelSpecs(
        name="Mistral 7B",
        parameters=7_241_732_096,
        layers=32,
        hidden_dim=4096,
        attention_heads=32,
        vocab_size=32000,
        context_length=8192,
        architecture="Mistral",
        fp32_size=29.0,
        fp16_size=14.5,
        int8_size=7.2,
        int4_size=3.6,
        relative_speed=1.1,  # Optimized architecture
        quality_score=72.0
    ),
    
    "Code-Llama-7B": ModelSpecs(
        name="Code Llama 7B",
        parameters=6_738_415_616,
        layers=32,
        hidden_dim=4096,
        attention_heads=32,
        vocab_size=32016,
        context_length=16384,
        architecture="Llama",
        fp32_size=26.9,
        fp16_size=13.5,
        int8_size=6.7,
        int4_size=3.4,
        relative_speed=1.0,
        quality_score=70.0,
        code_optimized=True
    ),
    
    # 13B Parameter Models
    "13B": ModelSpecs(
        name="Generic 13B",
        parameters=13_000_000_000,
        layers=40,
        hidden_dim=5120,
        attention_heads=40,
        vocab_size=32000,
        context_length=4096,
        architecture="Transformer",
        fp32_size=52.0,
        fp16_size=26.0,
        int8_size=13.0,
        int4_size=6.5,
        relative_speed=0.5,
        quality_score=75.0
    ),
    
    "Llama-2-13B": ModelSpecs(
        name="Llama 2 13B",
        parameters=13_015_864_320,
        layers=40,
        hidden_dim=5120,
        attention_heads=40,
        vocab_size=32000,
        context_length=4096,
        architecture="Llama",
        fp32_size=52.1,
        fp16_size=26.0,
        int8_size=13.0,
        int4_size=6.5,
        relative_speed=0.5,
        quality_score=78.0
    ),
    
    "Vicuna-13B": ModelSpecs(
        name="Vicuna 13B",
        parameters=13_015_864_320,
        layers=40,
        hidden_dim=5120,
        attention_heads=40,
        vocab_size=32000,
        context_length=2048,
        architecture="Llama",
        fp32_size=52.1,
        fp16_size=26.0,
        int8_size=13.0,
        int4_size=6.5,
        relative_speed=0.5,
        quality_score=76.0
    ),
    
    # 30B Parameter Models
    "30B": ModelSpecs(
        name="Generic 30B",
        parameters=30_000_000_000,
        layers=60,
        hidden_dim=6656,
        attention_heads=52,
        vocab_size=32000,
        context_length=4096,
        architecture="Transformer",
        fp32_size=120.0,
        fp16_size=60.0,
        int8_size=30.0,
        int4_size=15.0,
        relative_speed=0.2,
        quality_score=82.0
    ),
    
    # 65B Parameter Models
    "65B": ModelSpecs(
        name="Generic 65B",
        parameters=65_000_000_000,
        layers=80,
        hidden_dim=8192,
        attention_heads=64,
        vocab_size=32000,
        context_length=4096,
        architecture="Transformer",
        fp32_size=260.0,
        fp16_size=130.0,
        int8_size=65.0,
        int4_size=32.5,
        relative_speed=0.1,
        quality_score=85.0
    ),
    
    # Large Models (GPT-4 class)
    "GPT-4": ModelSpecs(
        name="GPT-4",
        parameters=1_760_000_000_000,  # Estimated
        layers=120,
        hidden_dim=12288,
        attention_heads=96,
        vocab_size=100000,
        context_length=8192,
        architecture="GPT",
        fp32_size=7040.0,
        fp16_size=3520.0,
        int8_size=1760.0,
        int4_size=880.0,
        relative_speed=0.01,
        quality_score=95.0,
        multimodal=True
    ),
    
    "Claude-3": ModelSpecs(
        name="Claude 3",
        parameters=500_000_000_000,  # Estimated
        layers=100,
        hidden_dim=10240,
        attention_heads=80,
        vocab_size=100000,
        context_length=200000,
        architecture="Transformer",
        fp32_size=2000.0,
        fp16_size=1000.0,
        int8_size=500.0,
        int4_size=250.0,
        relative_speed=0.03,
        quality_score=92.0,
        multimodal=True
    ),
    
    "PaLM-2": ModelSpecs(
        name="PaLM 2",
        parameters=340_000_000_000,  # Estimated
        layers=118,
        hidden_dim=18432,
        attention_heads=48,
        vocab_size=256000,
        context_length=8192,
        architecture="PaLM",
        fp32_size=1360.0,
        fp16_size=680.0,
        int8_size=340.0,
        int4_size=170.0,
        relative_speed=0.05,
        quality_score=90.0
    )
}


# Model categories for easy grouping
MODEL_CATEGORIES = {
    "small": ["7B", "Llama-2-7B", "Mistral-7B", "Code-Llama-7B"],
    "medium": ["13B", "Llama-2-13B", "Vicuna-13B"],
    "large": ["30B", "65B"],
    "xlarge": ["GPT-4", "Claude-3", "PaLM-2"]
}


# Benchmark scores for different models
BENCHMARK_SCORES = {
    "7B": {
        "mmlu": 45.3,
        "hellaswag": 77.2,
        "humaneval": 12.8,
        "gsm8k": 14.6,
        "truthfulqa": 38.5
    },
    "Llama-2-7B": {
        "mmlu": 45.3,
        "hellaswag": 77.2,
        "humaneval": 12.8,
        "gsm8k": 14.6,
        "truthfulqa": 38.5
    },
    "Mistral-7B": {
        "mmlu": 60.1,
        "hellaswag": 81.3,
        "humaneval": 29.8,
        "gsm8k": 52.2,
        "truthfulqa": 42.2
    },
    "Code-Llama-7B": {
        "mmlu": 35.1,
        "hellaswag": 74.1,
        "humaneval": 33.5,
        "gsm8k": 10.5,
        "truthfulqa": 37.8
    },
    "13B": {
        "mmlu": 56.8,
        "hellaswag": 80.9,
        "humaneval": 18.3,
        "gsm8k": 28.7,
        "truthfulqa": 41.9
    },
    "Llama-2-13B": {
        "mmlu": 54.8,
        "hellaswag": 80.9,
        "humaneval": 18.3,
        "gsm8k": 28.7,
        "truthfulqa": 41.9
    },
    "Vicuna-13B": {
        "mmlu": 52.1,
        "hellaswag": 79.2,
        "humaneval": 15.2,
        "gsm8k": 25.3,
        "truthfulqa": 44.1
    },
    "30B": {
        "mmlu": 62.5,
        "hellaswag": 84.2,
        "humaneval": 25.7,
        "gsm8k": 42.3,
        "truthfulqa": 45.8
    },
    "65B": {
        "mmlu": 68.9,
        "hellaswag": 87.3,
        "humaneval": 30.2,
        "gsm8k": 50.9,
        "truthfulqa": 47.9
    },
    "GPT-4": {
        "mmlu": 86.4,
        "hellaswag": 95.3,
        "humaneval": 67.0,
        "gsm8k": 92.0,
        "truthfulqa": 59.0
    },
    "Claude-3": {
        "mmlu": 84.9,
        "hellaswag": 94.1,
        "humaneval": 61.0,
        "gsm8k": 88.0,
        "truthfulqa": 83.0
    },
    "PaLM-2": {
        "mmlu": 78.5,
        "hellaswag": 89.7,
        "humaneval": 37.6,
        "gsm8k": 80.7,
        "truthfulqa": 64.0
    }
}


def get_model_specs(model_name: str) -> ModelSpecs:
    """Get model specifications by name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        ModelSpecs object
        
    Raises:
        ValueError: If model is not found
    """
    if model_name not in MODEL_DATABASE:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_DATABASE.keys())}")
    
    return MODEL_DATABASE[model_name]


def get_models_by_category(category: str) -> list[str]:
    """Get list of models in a category.
    
    Args:
        category: Category name (small, medium, large, xlarge)
        
    Returns:
        List of model names
        
    Raises:
        ValueError: If category is not found
    """
    if category not in MODEL_CATEGORIES:
        raise ValueError(f"Category '{category}' not found. Available categories: {list(MODEL_CATEGORIES.keys())}")
    
    return MODEL_CATEGORIES[category]


def get_benchmark_scores(model_name: str) -> Optional[Dict[str, float]]:
    """Get benchmark scores for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of benchmark scores or None if not available
    """
    return BENCHMARK_SCORES.get(model_name)


def compare_models(model_names: list[str], metric: str = "parameters") -> list[tuple[str, float]]:
    """Compare models by a specific metric.
    
    Args:
        model_names: List of model names to compare
        metric: Metric to compare (parameters, quality_score, relative_speed, etc.)
        
    Returns:
        List of (model_name, metric_value) tuples sorted by metric
    """
    results = []
    
    for model_name in model_names:
        try:
            model_spec = get_model_specs(model_name)
            if hasattr(model_spec, metric):
                value = getattr(model_spec, metric)
                results.append((model_name, value))
        except ValueError:
            continue
    
    # Sort by metric value (descending for most metrics)
    reverse = metric not in ["relative_speed"]  # Speed is inverse (lower is better)
    results.sort(key=lambda x: x[1], reverse=reverse)
    
    return results


def estimate_model_size(parameters: int, precision: str = "fp16") -> float:
    """Estimate model size in GB based on parameter count.
    
    Args:
        parameters: Number of parameters
        precision: Precision mode (fp32, fp16, int8, int4)
        
    Returns:
        Estimated size in GB
    """
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "int8": 1,
        "int4": 0.5
    }
    
    if precision not in bytes_per_param:
        raise ValueError(f"Unknown precision '{precision}'. Available: {list(bytes_per_param.keys())}")
    
    size_bytes = parameters * bytes_per_param[precision]
    size_gb = size_bytes / (1024 ** 3)
    
    return size_gb


def get_all_models() -> list[str]:
    """Get list of all available models.
    
    Returns:
        List of model names
    """
    return list(MODEL_DATABASE.keys())


def get_model_info(model_name: str) -> Dict:
    """Get comprehensive information about a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model specs and benchmark scores
    """
    model_spec = get_model_specs(model_name)
    benchmark_scores = get_benchmark_scores(model_name)
    
    info = {
        "specs": model_spec,
        "benchmarks": benchmark_scores,
        "category": None
    }
    
    # Find category
    for category, models in MODEL_CATEGORIES.items():
        if model_name in models:
            info["category"] = category
            break
    
    return info