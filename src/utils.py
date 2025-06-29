#!/usr/bin/env python3
"""
Utility Functions

Helper functions for validation, formatting, and common operations.
"""

import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict
import json


def validate_inputs(
    model_size: str,
    tokens: int,
    batch_size: int,
    hardware_type: str,
    deployment_mode: str,
    precision_mode: str
) -> None:
    """Validate calculator inputs.
    
    Args:
        model_size: Model size identifier
        tokens: Number of tokens
        batch_size: Batch size
        hardware_type: Hardware identifier
        deployment_mode: Deployment mode
        precision_mode: Precision mode
        
    Raises:
        ValueError: If any input is invalid
    """
    # Validate model_size
    if not isinstance(model_size, str) or not model_size.strip():
        raise ValueError("model_size must be a non-empty string")
    
    # Validate tokens
    if not isinstance(tokens, int) or tokens <= 0:
        raise ValueError("tokens must be a positive integer")
    
    if tokens > 1_000_000:
        raise ValueError("tokens cannot exceed 1,000,000")
    
    # Validate batch_size
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    
    if batch_size > 1000:
        raise ValueError("batch_size cannot exceed 1000")
    
    # Validate hardware_type
    if not isinstance(hardware_type, str) or not hardware_type.strip():
        raise ValueError("hardware_type must be a non-empty string")
    
    # Validate deployment_mode
    valid_deployment_modes = ["local", "cloud", "edge", "serverless"]
    if deployment_mode not in valid_deployment_modes:
        raise ValueError(f"deployment_mode must be one of {valid_deployment_modes}")
    
    # Validate precision_mode
    valid_precision_modes = ["fp32", "fp16", "int8", "int4"]
    if precision_mode not in valid_precision_modes:
        raise ValueError(f"precision_mode must be one of {valid_precision_modes}")


def format_memory(memory_gb: float) -> str:
    """Format memory size in human-readable format.
    
    Args:
        memory_gb: Memory size in GB
        
    Returns:
        Formatted memory string
    """
    if memory_gb < 1.0:
        return f"{memory_gb * 1024:.1f} MB"
    elif memory_gb < 1024.0:
        return f"{memory_gb:.1f} GB"
    else:
        return f"{memory_gb / 1024:.1f} TB"


def format_latency(latency_ms: float) -> str:
    """Format latency in human-readable format.
    
    Args:
        latency_ms: Latency in milliseconds
        
    Returns:
        Formatted latency string
    """
    if latency_ms < 1000:
        return f"{latency_ms:.1f} ms"
    elif latency_ms < 60000:
        return f"{latency_ms / 1000:.1f} s"
    else:
        return f"{latency_ms / 60000:.1f} min"


def format_cost(cost_usd: float) -> str:
    """Format cost in human-readable format.
    
    Args:
        cost_usd: Cost in USD
        
    Returns:
        Formatted cost string
    """
    if cost_usd < 0.01:
        return f"${cost_usd * 1000:.2f}m"  # millidollars
    elif cost_usd < 1.0:
        return f"${cost_usd:.3f}"
    else:
        return f"${cost_usd:.2f}"


def format_throughput(throughput: float, unit: str = "tokens/s") -> str:
    """Format throughput in human-readable format.
    
    Args:
        throughput: Throughput value
        unit: Unit string
        
    Returns:
        Formatted throughput string
    """
    if throughput < 1.0:
        return f"{throughput:.2f} {unit}"
    elif throughput < 1000:
        return f"{throughput:.1f} {unit}"
    else:
        return f"{throughput / 1000:.1f} k{unit}"


def format_percentage(value: float) -> str:
    """Format percentage value.
    
    Args:
        value: Percentage value (0.0 to 1.0)
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.1f}%"


def format_results(result) -> str:
    """Format calculation results for display.
    
    Args:
        result: InferenceResult object
        
    Returns:
        Formatted results string
    """
    compatibility_status = "✅ Compatible" if result.hardware_compatible else "❌ Incompatible"
    
    output = f"""
🔍 LLM Inference Analysis Results
{'=' * 50}

📊 Model Configuration:
  • Model: {result.model_size}
  • Precision: {result.precision_mode.upper()}
  • Input tokens: {result.input_tokens:,}
  • Output tokens: {result.output_tokens:,}
  • Batch size: {result.batch_size}

🖥️  Hardware Configuration:
  • Hardware: {result.hardware_type}
  • Deployment: {result.deployment_mode}
  • Compatibility: {compatibility_status}
  • Memory utilization: {format_percentage(result.memory_utilization)}

⚡ Performance Metrics:
  • Total latency: {format_latency(result.latency_ms)}
  • Time to first token: {format_latency(result.ttft_ms)}
  • Time per output token: {format_latency(result.tpot_ms)}
  • Tokens per second: {format_throughput(result.tokens_per_second)}
  • Requests per second: {format_throughput(result.requests_per_second, "req/s")}

💾 Memory Usage:
  • Total memory: {format_memory(result.memory_gb)}
  • Model memory: {format_memory(result.model_memory_gb)}
  • KV cache: {format_memory(result.kv_cache_gb)}
  • Activations: {format_memory(result.activation_memory_gb)}

💰 Cost Analysis:
  • Cost per request: {format_cost(result.cost_per_request)}
  • Cost per token: {format_cost(result.cost_per_token)}
  • Cost per hour: {format_cost(result.cost_per_hour)}
"""
    
    if not result.hardware_compatible:
        output += f"""

⚠️  Compatibility Warning:
The selected hardware may not have sufficient memory to run this model efficiently.
Consider:
  • Using quantization (INT8/INT4)
  • Upgrading to hardware with more memory
  • Reducing batch size
  • Using a smaller model
"""
    
    return output


def parse_model_size(model_size_str: str) -> int:
    """Parse model size string to parameter count.
    
    Args:
        model_size_str: Model size string (e.g., "7B", "13B", "1.3T")
        
    Returns:
        Number of parameters
        
    Raises:
        ValueError: If format is invalid
    """
    # Remove whitespace and convert to uppercase
    size_str = model_size_str.strip().upper()
    
    # Match pattern like "7B", "13B", "1.3T"
    pattern = r'^(\d+(?:\.\d+)?)([KMBT])$'
    match = re.match(pattern, size_str)
    
    if not match:
        raise ValueError(f"Invalid model size format: {model_size_str}")
    
    number, suffix = match.groups()
    number = float(number)
    
    multipliers = {
        'K': 1_000,
        'M': 1_000_000,
        'B': 1_000_000_000,
        'T': 1_000_000_000_000
    }
    
    if suffix not in multipliers:
        raise ValueError(f"Invalid size suffix: {suffix}")
    
    return int(number * multipliers[suffix])


def format_model_size(parameters: int) -> str:
    """Format parameter count to human-readable string.
    
    Args:
        parameters: Number of parameters
        
    Returns:
        Formatted size string
    """
    if parameters >= 1_000_000_000_000:
        return f"{parameters / 1_000_000_000_000:.1f}T"
    elif parameters >= 1_000_000_000:
        return f"{parameters / 1_000_000_000:.1f}B"
    elif parameters >= 1_000_000:
        return f"{parameters / 1_000_000:.1f}M"
    elif parameters >= 1_000:
        return f"{parameters / 1_000:.1f}K"
    else:
        return str(parameters)


def calculate_roi(
    initial_cost: float,
    operational_cost_per_hour: float,
    revenue_per_hour: float,
    time_horizon_hours: float
) -> Dict[str, float]:
    """Calculate return on investment for hardware.
    
    Args:
        initial_cost: Initial hardware cost
        operational_cost_per_hour: Operational cost per hour
        revenue_per_hour: Revenue per hour
        time_horizon_hours: Time horizon in hours
        
    Returns:
        Dictionary with ROI metrics
    """
    total_operational_cost = operational_cost_per_hour * time_horizon_hours
    total_revenue = revenue_per_hour * time_horizon_hours
    total_cost = initial_cost + total_operational_cost
    
    profit = total_revenue - total_cost
    roi = (profit / total_cost) * 100 if total_cost > 0 else 0
    
    # Break-even time
    net_hourly_profit = revenue_per_hour - operational_cost_per_hour
    breakeven_hours = initial_cost / net_hourly_profit if net_hourly_profit > 0 else float('inf')
    
    return {
        "total_cost": total_cost,
        "total_revenue": total_revenue,
        "profit": profit,
        "roi_percentage": roi,
        "breakeven_hours": breakeven_hours,
        "breakeven_days": breakeven_hours / 24
    }


def estimate_scaling_requirements(
    current_rps: float,
    target_rps: float,
    current_hardware_count: int = 1
) -> Dict[str, Union[int, float]]:
    """Estimate scaling requirements for increased load.
    
    Args:
        current_rps: Current requests per second
        target_rps: Target requests per second
        current_hardware_count: Current number of hardware units
        
    Returns:
        Dictionary with scaling recommendations
    """
    if current_rps <= 0:
        raise ValueError("current_rps must be positive")
    
    scaling_factor = target_rps / current_rps
    
    # Account for efficiency loss in scaling
    efficiency_factor = 0.85  # Assume 15% efficiency loss
    required_hardware = int(scaling_factor * current_hardware_count / efficiency_factor) + 1
    
    actual_capacity = required_hardware * current_rps * efficiency_factor
    overhead_percentage = (actual_capacity - target_rps) / target_rps * 100
    
    return {
        "required_hardware_units": required_hardware,
        "scaling_factor": scaling_factor,
        "efficiency_factor": efficiency_factor,
        "actual_capacity_rps": actual_capacity,
        "overhead_percentage": overhead_percentage
    }


def generate_summary_report(results: List[Any]) -> str:
    """Generate a summary report from multiple calculation results.
    
    Args:
        results: List of InferenceResult objects
        
    Returns:
        Formatted summary report string
    """
    if not results:
        return "No results to summarize."
    
    report = ["# LLM Inference Calculator Summary Report\n"]
    
    # Overview
    report.append(f"**Total Configurations Analyzed:** {len(results)}\n")
    
    # Find best options
    compatible_results = [r for r in results if r.hardware_compatible]
    
    if not compatible_results:
        report.append("⚠️ **Warning:** No compatible hardware configurations found.\n")
        return "\n".join(report)
    
    # Best latency
    best_latency = min(compatible_results, key=lambda x: x.latency_ms)
    report.append(f"**Lowest Latency:** {best_latency.hardware_type} - {format_latency(best_latency.latency_ms)}")
    
    # Best cost
    best_cost = min(compatible_results, key=lambda x: x.cost_per_request)
    report.append(f"**Lowest Cost:** {best_cost.hardware_type} - {format_cost(best_cost.cost_per_request)}/request")
    
    # Best throughput
    best_throughput = max(compatible_results, key=lambda x: x.tokens_per_second)
    report.append(f"**Highest Throughput:** {best_throughput.hardware_type} - {format_throughput(best_throughput.tokens_per_second)}")
    
    # Memory requirements
    min_memory = min(compatible_results, key=lambda x: x.memory_gb)
    max_memory = max(compatible_results, key=lambda x: x.memory_gb)
    report.append(f"\n**Memory Requirements:** {format_memory(min_memory.memory_gb)} - {format_memory(max_memory.memory_gb)}")
    
    # Cost range
    min_cost = min(compatible_results, key=lambda x: x.cost_per_hour)
    max_cost = max(compatible_results, key=lambda x: x.cost_per_hour)
    report.append(f"**Cost Range:** {format_cost(min_cost.cost_per_hour)}/hour - {format_cost(max_cost.cost_per_hour)}/hour")
    
    return "\n".join(report)


def export_results_json(results: Union[Any, List[Any]], filename: str) -> None:
    """Export results to JSON file.
    
    Args:
        results: Single result or list of results
        filename: Output filename
    """
    if not isinstance(results, list):
        results = [results]
    
    # Convert dataclass objects to dictionaries
    json_data = []
    for result in results:
        if hasattr(result, '__dict__'):
            json_data.append(asdict(result))
        else:
            json_data.append(result)
    
    with open(filename, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)


def load_results_json(filename: str) -> List[Dict[str, Any]]:
    """Load results from JSON file.
    
    Args:
        filename: Input filename
        
    Returns:
        List of result dictionaries
    """
    with open(filename, 'r') as f:
        return json.load(f)


def validate_hardware_compatibility(
    model_memory_gb: float,
    hardware_memory_gb: float,
    safety_margin: float = 0.1
) -> Dict[str, Union[bool, float, str]]:
    """Validate if hardware can run the model.
    
    Args:
        model_memory_gb: Required model memory
        hardware_memory_gb: Available hardware memory
        safety_margin: Safety margin (10% by default)
        
    Returns:
        Dictionary with compatibility information
    """
    required_memory = model_memory_gb * (1 + safety_margin)
    compatible = hardware_memory_gb >= required_memory
    utilization = model_memory_gb / hardware_memory_gb
    
    if not compatible:
        status = "Insufficient memory"
    elif utilization > 0.9:
        status = "High memory usage - may cause OOM"
    elif utilization > 0.7:
        status = "Moderate memory usage"
    else:
        status = "Comfortable memory headroom"
    
    return {
        "compatible": compatible,
        "memory_utilization": utilization,
        "required_memory_gb": required_memory,
        "available_memory_gb": hardware_memory_gb,
        "status": status
    }


def calculate_carbon_footprint(
    power_watts: float,
    hours: float,
    carbon_intensity_g_per_kwh: float = 400.0
) -> Dict[str, float]:
    """Calculate carbon footprint of hardware usage.
    
    Args:
        power_watts: Power consumption in watts
        hours: Usage time in hours
        carbon_intensity_g_per_kwh: Carbon intensity (grams CO2 per kWh)
        
    Returns:
        Dictionary with carbon footprint metrics
    """
    energy_kwh = (power_watts / 1000) * hours
    carbon_grams = energy_kwh * carbon_intensity_g_per_kwh
    carbon_kg = carbon_grams / 1000
    
    return {
        "energy_kwh": energy_kwh,
        "carbon_grams": carbon_grams,
        "carbon_kg": carbon_kg,
        "carbon_tons": carbon_kg / 1000
    }


def optimize_batch_size_for_latency(
    base_latency_ms: float,
    target_latency_ms: float,
    current_batch_size: int = 1
) -> int:
    """Optimize batch size to meet latency target.
    
    Args:
        base_latency_ms: Base latency for single request
        target_latency_ms: Target latency
        current_batch_size: Current batch size
        
    Returns:
        Recommended batch size
    """
    if target_latency_ms < base_latency_ms:
        return 1  # Cannot go below base latency
    
    # Estimate batch size based on linear scaling assumption
    max_batch_size = int(target_latency_ms / base_latency_ms)
    
    # Apply efficiency factor (batching has diminishing returns)
    efficiency_factor = 0.8
    recommended_batch_size = int(max_batch_size * efficiency_factor)
    
    return max(1, recommended_batch_size)