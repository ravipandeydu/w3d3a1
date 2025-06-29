#!/usr/bin/env python3
"""
LLM Inference Calculator Package

A comprehensive calculator for estimating LLM inference costs, latency, and memory usage.
"""

from .calculator import LLMInferenceCalculator, InferenceResult, DeploymentMode, PrecisionMode
from .models import ModelSpecs, get_model_specs, get_models_by_category, get_all_models, MODEL_DATABASE
from .hardware import HardwareSpecs, get_hardware_specs, get_hardware_by_category, get_all_hardware, HARDWARE_DATABASE
from .utils import (
    validate_inputs,
    format_memory,
    format_latency,
    format_cost,
    format_throughput,
    format_results,
    generate_summary_report
)

__version__ = "1.0.0"
__author__ = "LLM Inference Calculator Team"
__description__ = "Calculator for estimating LLM inference costs, latency, and memory usage"

__all__ = [
    # Main calculator
    "LLMInferenceCalculator",
    "InferenceResult",
    "DeploymentMode",
    "PrecisionMode",
    
    # Model specifications
    "ModelSpecs",
    "get_model_specs",
    "get_models_by_category",
    "get_all_models",
    "MODEL_DATABASE",
    
    # Hardware specifications
    "HardwareSpecs",
    "get_hardware_specs",
    "get_hardware_by_category",
    "get_all_hardware",
    "HARDWARE_DATABASE",
    
    # Utilities
    "validate_inputs",
    "format_memory",
    "format_latency",
    "format_cost",
    "format_throughput",
    "format_results",
    "generate_summary_report"
]