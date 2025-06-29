#!/usr/bin/env python3
"""
LLM Inference Calculator

Main calculator implementation for estimating LLM inference costs, latency, and memory usage.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .models import ModelSpecs, get_model_specs
from .hardware import HardwareSpecs, get_hardware_specs
from .utils import validate_inputs, format_results


class DeploymentMode(Enum):
    """Supported deployment modes."""
    LOCAL = "local"
    CLOUD = "cloud"
    EDGE = "edge"
    SERVERLESS = "serverless"


class PrecisionMode(Enum):
    """Supported precision modes."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class InferenceResult:
    """Result of inference calculation."""
    # Latency metrics
    latency_ms: float
    ttft_ms: float  # Time to first token
    tpot_ms: float  # Time per output token
    
    # Memory metrics
    memory_gb: float
    model_memory_gb: float
    kv_cache_gb: float
    activation_memory_gb: float
    
    # Cost metrics
    cost_per_request: float
    cost_per_token: float
    cost_per_hour: float
    
    # Performance metrics
    tokens_per_second: float
    requests_per_second: float
    
    # Compatibility
    hardware_compatible: bool
    memory_utilization: float
    
    # Configuration
    model_size: str
    hardware_type: str
    deployment_mode: str
    precision_mode: str
    batch_size: int
    input_tokens: int
    output_tokens: int


class LLMInferenceCalculator:
    """Main calculator for LLM inference metrics."""
    
    def __init__(self):
        """Initialize the calculator."""
        self.model_specs = {}
        self.hardware_specs = {}
        self._load_specifications()
    
    def _load_specifications(self):
        """Load model and hardware specifications."""
        # Load model specifications
        model_names = [
            "7B", "13B", "30B", "65B", "GPT-4",
            "Llama-2-7B", "Llama-2-13B", "Mistral-7B",
            "Code-Llama-7B", "Vicuna-13B"
        ]
        
        for name in model_names:
            try:
                self.model_specs[name] = get_model_specs(name)
            except ValueError:
                continue
        
        # Load hardware specifications
        hardware_names = [
            "RTX_3080", "RTX_3090", "RTX_4080", "RTX_4090", "GTX_1650",
            "A100_40GB", "A100_80GB", "H100", "V100",
            "CPU_16GB", "CPU_32GB", "CPU_64GB",
            "M1_Pro", "M1_Max", "M2_Ultra"
        ]
        
        for name in hardware_names:
            try:
                self.hardware_specs[name] = get_hardware_specs(name)
            except ValueError:
                continue
    
    def calculate(
        self,
        model_size: str,
        tokens: int,
        batch_size: int = 1,
        hardware_type: str = "GTX_1650",
        deployment_mode: str = "local",
        precision_mode: str = "fp16",
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None
    ) -> InferenceResult:
        """
        Calculate inference metrics for given parameters.
        
        Args:
            model_size: Model size identifier (e.g., "7B", "13B", "GPT-4")
            tokens: Total number of tokens (input + output)
            batch_size: Number of concurrent requests
            hardware_type: Hardware identifier
            deployment_mode: Deployment mode
            precision_mode: Model precision
            input_tokens: Number of input tokens (defaults to tokens/2)
            output_tokens: Number of output tokens (defaults to tokens/2)
        
        Returns:
            InferenceResult with all calculated metrics
        """
        # Validate inputs
        validate_inputs(
            model_size, tokens, batch_size, hardware_type,
            deployment_mode, precision_mode
        )
        
        # Set default token split
        if input_tokens is None and output_tokens is None:
            input_tokens = tokens // 2
            output_tokens = tokens - input_tokens
        elif input_tokens is None:
            input_tokens = tokens - output_tokens
        elif output_tokens is None:
            output_tokens = tokens - input_tokens
        
        # Get specifications
        model_spec = self.model_specs[model_size]
        hardware_spec = self.hardware_specs[hardware_type]
        
        # Calculate memory requirements
        memory_metrics = self._calculate_memory(
            model_spec, hardware_spec, batch_size,
            input_tokens, output_tokens, precision_mode
        )
        
        # Calculate latency
        latency_metrics = self._calculate_latency(
            model_spec, hardware_spec, batch_size,
            input_tokens, output_tokens, precision_mode
        )
        
        # Calculate costs
        cost_metrics = self._calculate_costs(
            hardware_spec, deployment_mode, batch_size,
            input_tokens + output_tokens, latency_metrics["total_latency"]
        )
        
        # Calculate performance
        performance_metrics = self._calculate_performance(
            latency_metrics, batch_size, output_tokens
        )
        
        # Check compatibility
        compatibility = self._check_compatibility(
            memory_metrics, hardware_spec
        )
        
        return InferenceResult(
            # Latency
            latency_ms=latency_metrics["total_latency"],
            ttft_ms=latency_metrics["ttft"],
            tpot_ms=latency_metrics["tpot"],
            
            # Memory
            memory_gb=memory_metrics["total_memory"],
            model_memory_gb=memory_metrics["model_memory"],
            kv_cache_gb=memory_metrics["kv_cache"],
            activation_memory_gb=memory_metrics["activation_memory"],
            
            # Cost
            cost_per_request=cost_metrics["cost_per_request"],
            cost_per_token=cost_metrics["cost_per_token"],
            cost_per_hour=cost_metrics["cost_per_hour"],
            
            # Performance
            tokens_per_second=performance_metrics["tokens_per_second"],
            requests_per_second=performance_metrics["requests_per_second"],
            
            # Compatibility
            hardware_compatible=compatibility["compatible"],
            memory_utilization=compatibility["memory_utilization"],
            
            # Configuration
            model_size=model_size,
            hardware_type=hardware_type,
            deployment_mode=deployment_mode,
            precision_mode=precision_mode,
            batch_size=batch_size,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
    
    def _calculate_memory(
        self,
        model_spec: ModelSpecs,
        hardware_spec: HardwareSpecs,
        batch_size: int,
        input_tokens: int,
        output_tokens: int,
        precision_mode: str
    ) -> Dict[str, float]:
        """Calculate memory requirements."""
        # Precision multipliers (bytes per parameter)
        precision_multipliers = {
            "fp32": 4.0,
            "fp16": 2.0,
            "int8": 1.0,
            "int4": 0.5
        }
        
        precision_mult = precision_multipliers[precision_mode]
        
        # Model memory (parameters)
        model_memory = (model_spec.parameters * precision_mult) / (1024**3)  # GB
        
        # KV cache memory
        # Each token stores key and value vectors
        # Memory = batch_size * sequence_length * 2 * hidden_dim * precision
        sequence_length = input_tokens + output_tokens
        kv_cache = (
            batch_size * sequence_length * 2 * 
            model_spec.hidden_dim * precision_mult
        ) / (1024**3)
        
        # Activation memory (rough estimate)
        # Depends on batch size, sequence length, and model architecture
        activation_memory = (
            batch_size * sequence_length * model_spec.hidden_dim * 
            precision_mult * 4  # Factor for intermediate activations
        ) / (1024**3)
        
        # Overhead (framework, buffers, etc.)
        overhead = max(2.0, model_memory * 0.1)  # At least 2GB or 10% of model
        
        total_memory = model_memory + kv_cache + activation_memory + overhead
        
        return {
            "model_memory": model_memory,
            "kv_cache": kv_cache,
            "activation_memory": activation_memory,
            "overhead": overhead,
            "total_memory": total_memory
        }
    
    def _calculate_latency(
        self,
        model_spec: ModelSpecs,
        hardware_spec: HardwareSpecs,
        batch_size: int,
        input_tokens: int,
        output_tokens: int,
        precision_mode: str
    ) -> Dict[str, float]:
        """Calculate latency metrics."""
        # Base latency factors (ms per token per billion parameters)
        base_latency_factors = {
            "fp32": 2.0,
            "fp16": 1.0,
            "int8": 0.7,
            "int4": 0.5
        }
        
        base_factor = base_latency_factors[precision_mode]
        param_billions = model_spec.parameters / 1e9
        
        # Hardware performance multiplier
        hw_multiplier = hardware_spec.performance_multiplier
        
        # Time to First Token (prefill phase)
        # Depends on input length and model size
        ttft = (
            input_tokens * param_billions * base_factor * hw_multiplier
        )
        
        # Time Per Output Token (decode phase)
        # Includes KV cache overhead
        sequence_length = input_tokens + output_tokens
        kv_overhead = 1 + (sequence_length / 2048) * 0.1  # 10% overhead per 2K tokens
        
        tpot = (
            param_billions * base_factor * hw_multiplier * kv_overhead
        )
        
        # Batch processing efficiency
        if batch_size > 1:
            # Batching improves efficiency but adds some overhead
            batch_efficiency = min(0.9, 0.5 + 0.4 * math.log2(batch_size))
            ttft *= batch_efficiency
            tpot *= batch_efficiency
        
        # Total latency
        total_latency = ttft + (output_tokens * tpot)
        
        return {
            "ttft": ttft,
            "tpot": tpot,
            "total_latency": total_latency
        }
    
    def _calculate_costs(
        self,
        hardware_spec: HardwareSpecs,
        deployment_mode: str,
        batch_size: int,
        total_tokens: int,
        latency_ms: float
    ) -> Dict[str, float]:
        """Calculate cost metrics."""
        # Cost factors by deployment mode
        deployment_costs = {
            "local": {
                "hardware_cost_per_hour": hardware_spec.cost_per_hour_local,
                "power_cost_per_hour": hardware_spec.power_watts * 0.12 / 1000,  # $0.12/kWh
                "overhead_multiplier": 1.2  # 20% overhead
            },
            "cloud": {
                "hardware_cost_per_hour": hardware_spec.cost_per_hour_cloud,
                "power_cost_per_hour": 0,  # Included in cloud pricing
                "overhead_multiplier": 1.1  # 10% overhead
            },
            "edge": {
                "hardware_cost_per_hour": hardware_spec.cost_per_hour_local * 0.5,
                "power_cost_per_hour": hardware_spec.power_watts * 0.15 / 1000,  # Higher edge power cost
                "overhead_multiplier": 1.5  # 50% overhead for edge complexity
            },
            "serverless": {
                "hardware_cost_per_hour": hardware_spec.cost_per_hour_cloud * 2,
                "power_cost_per_hour": 0,
                "overhead_multiplier": 1.0  # Pay per use
            }
        }
        
        costs = deployment_costs[deployment_mode]
        
        # Total cost per hour
        cost_per_hour = (
            costs["hardware_cost_per_hour"] + 
            costs["power_cost_per_hour"]
        ) * costs["overhead_multiplier"]
        
        # Cost per request
        request_time_hours = latency_ms / (1000 * 3600)  # Convert ms to hours
        cost_per_request = cost_per_hour * request_time_hours / batch_size
        
        # Cost per token
        cost_per_token = cost_per_request / total_tokens if total_tokens > 0 else 0
        
        return {
            "cost_per_hour": cost_per_hour,
            "cost_per_request": cost_per_request,
            "cost_per_token": cost_per_token
        }
    
    def _calculate_performance(
        self,
        latency_metrics: Dict[str, float],
        batch_size: int,
        output_tokens: int
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        tpot_ms = latency_metrics["tpot"]
        total_latency_ms = latency_metrics["total_latency"]
        
        # Tokens per second (for decode phase)
        tokens_per_second = 1000 / tpot_ms if tpot_ms > 0 else 0
        
        # Requests per second
        requests_per_second = (
            1000 * batch_size / total_latency_ms if total_latency_ms > 0 else 0
        )
        
        return {
            "tokens_per_second": tokens_per_second,
            "requests_per_second": requests_per_second
        }
    
    def _check_compatibility(
        self,
        memory_metrics: Dict[str, float],
        hardware_spec: HardwareSpecs
    ) -> Dict[str, float]:
        """Check hardware compatibility."""
        required_memory = memory_metrics["total_memory"]
        available_memory = hardware_spec.memory_gb
        
        compatible = required_memory <= available_memory
        memory_utilization = required_memory / available_memory if available_memory > 0 else float('inf')
        
        return {
            "compatible": compatible,
            "memory_utilization": memory_utilization
        }
    
    def compare_models(
        self,
        model_sizes: List[str],
        tokens: int,
        hardware_type: str = "GTX_1650",
        **kwargs
    ) -> List[InferenceResult]:
        """Compare multiple models with the same parameters."""
        results = []
        for model_size in model_sizes:
            try:
                result = self.calculate(
                    model_size=model_size,
                    tokens=tokens,
                    hardware_type=hardware_type,
                    **kwargs
                )
                results.append(result)
            except (KeyError, ValueError) as e:
                print(f"Warning: Could not calculate for {model_size}: {e}")
        return results
    
    def compare_hardware(
        self,
        hardware_types: List[str],
        model_size: str,
        tokens: int,
        **kwargs
    ) -> List[InferenceResult]:
        """Compare multiple hardware options with the same model."""
        results = []
        for hardware_type in hardware_types:
            try:
                result = self.calculate(
                    model_size=model_size,
                    tokens=tokens,
                    hardware_type=hardware_type,
                    **kwargs
                )
                results.append(result)
            except (KeyError, ValueError) as e:
                print(f"Warning: Could not calculate for {hardware_type}: {e}")
        return results
    
    def optimize_batch_size(
        self,
        model_size: str,
        tokens: int,
        hardware_type: str,
        max_batch_size: int = 32,
        **kwargs
    ) -> Tuple[int, InferenceResult]:
        """Find optimal batch size for maximum throughput."""
        best_batch_size = 1
        best_throughput = 0
        best_result = None
        
        for batch_size in range(1, max_batch_size + 1):
            try:
                result = self.calculate(
                    model_size=model_size,
                    tokens=tokens,
                    hardware_type=hardware_type,
                    batch_size=batch_size,
                    **kwargs
                )
                
                if not result.hardware_compatible:
                    break
                
                throughput = result.requests_per_second
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = batch_size
                    best_result = result
                    
            except (KeyError, ValueError):
                break
        
        return best_batch_size, best_result
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported model sizes."""
        return list(self.model_specs.keys())
    
    def get_supported_hardware(self) -> List[str]:
        """Get list of supported hardware types."""
        return list(self.hardware_specs.keys())