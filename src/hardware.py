#!/usr/bin/env python3
"""
Hardware Specifications

Defines specifications and parameters for different hardware configurations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class HardwareType(Enum):
    """Types of hardware."""
    CONSUMER_GPU = "consumer_gpu"
    PROFESSIONAL_GPU = "professional_gpu"
    CPU = "cpu"
    MOBILE = "mobile"
    CLOUD = "cloud"


@dataclass
class HardwareSpecs:
    """Specifications for hardware configuration."""
    name: str
    hardware_type: HardwareType
    
    # Memory specifications
    memory_gb: float
    memory_bandwidth_gbps: float
    memory_type: str
    
    # Compute specifications
    compute_units: int
    base_clock_mhz: int
    boost_clock_mhz: int
    fp16_tflops: float
    fp32_tflops: float
    int8_tops: float
    
    # Power and thermal
    power_watts: int
    tdp_watts: int
    
    # Performance characteristics
    performance_multiplier: float  # Relative to baseline
    efficiency_score: float  # Performance per watt
    
    # Cost information
    msrp_usd: int
    cost_per_hour_local: float  # Amortized cost
    cost_per_hour_cloud: float  # Cloud instance cost
    
    # Compatibility
    supports_fp16: bool = True
    supports_int8: bool = True
    supports_int4: bool = False
    tensor_cores: bool = False
    
    # Additional features
    nvlink: bool = False
    multi_gpu_scaling: float = 1.0  # Efficiency when using multiple GPUs


# Hardware database
HARDWARE_DATABASE: Dict[str, HardwareSpecs] = {
    # Consumer GPUs - NVIDIA RTX Series
    "RTX_3080": HardwareSpecs(
        name="NVIDIA RTX 3080",
        hardware_type=HardwareType.CONSUMER_GPU,
        memory_gb=10.0,
        memory_bandwidth_gbps=760.0,
        memory_type="GDDR6X",
        compute_units=68,
        base_clock_mhz=1440,
        boost_clock_mhz=1710,
        fp16_tflops=59.7,
        fp32_tflops=29.8,
        int8_tops=119.4,
        power_watts=320,
        tdp_watts=320,
        performance_multiplier=0.8,
        efficiency_score=0.093,  # TFLOPS/Watt
        msrp_usd=699,
        cost_per_hour_local=0.039,  # Based on 3-year amortization
        cost_per_hour_cloud=0.90,
        tensor_cores=True,
        supports_int4=True
    ),
    
    "RTX_3090": HardwareSpecs(
        name="NVIDIA RTX 3090",
        hardware_type=HardwareType.CONSUMER_GPU,
        memory_gb=24.0,
        memory_bandwidth_gbps=936.0,
        memory_type="GDDR6X",
        compute_units=82,
        base_clock_mhz=1395,
        boost_clock_mhz=1695,
        fp16_tflops=71.0,
        fp32_tflops=35.5,
        int8_tops=142.0,
        power_watts=350,
        tdp_watts=350,
        performance_multiplier=0.9,
        efficiency_score=0.101,
        msrp_usd=1499,
        cost_per_hour_local=0.083,
        cost_per_hour_cloud=1.20,
        tensor_cores=True,
        supports_int4=True
    ),
    
    "RTX_4080": HardwareSpecs(
        name="NVIDIA RTX 4080",
        hardware_type=HardwareType.CONSUMER_GPU,
        memory_gb=16.0,
        memory_bandwidth_gbps=717.0,
        memory_type="GDDR6X",
        compute_units=76,
        base_clock_mhz=2205,
        boost_clock_mhz=2505,
        fp16_tflops=97.4,
        fp32_tflops=48.7,
        int8_tops=194.8,
        power_watts=320,
        tdp_watts=320,
        performance_multiplier=0.7,
        efficiency_score=0.152,
        msrp_usd=1199,
        cost_per_hour_local=0.067,
        cost_per_hour_cloud=1.10,
        tensor_cores=True,
        supports_int4=True
    ),
    
    "RTX_4090": HardwareSpecs(
        name="NVIDIA RTX 4090",
        hardware_type=HardwareType.CONSUMER_GPU,
        memory_gb=24.0,
        memory_bandwidth_gbps=1008.0,
        memory_type="GDDR6X",
        compute_units=128,
        base_clock_mhz=2230,
        boost_clock_mhz=2520,
        fp16_tflops=165.2,
        fp32_tflops=82.6,
        int8_tops=330.4,
        power_watts=450,
        tdp_watts=450,
        performance_multiplier=0.6,
        efficiency_score=0.184,
        msrp_usd=1599,
        cost_per_hour_local=0.089,
        cost_per_hour_cloud=1.50,
        tensor_cores=True,
        supports_int4=True
    ),
    
    # GTX Series - Older Consumer GPUs
    "GTX_1650": HardwareSpecs(
        name="NVIDIA GTX 1650",
        hardware_type=HardwareType.CONSUMER_GPU,
        memory_gb=4.0,
        memory_bandwidth_gbps=128.0,
        memory_type="GDDR5",
        compute_units=14,
        base_clock_mhz=1485,
        boost_clock_mhz=1665,
        fp16_tflops=4.1,
        fp32_tflops=2.9,
        int8_tops=8.2,
        power_watts=75,
        tdp_watts=75,
        performance_multiplier=3.5,  # Slower for LLM inference
        efficiency_score=0.039,  # TFLOPS/Watt
        msrp_usd=149,
        cost_per_hour_local=0.008,  # Based on 3-year amortization
        cost_per_hour_cloud=0.25,
        tensor_cores=False,
        supports_int4=False,
        supports_int8=True
    ),
    
    # Professional GPUs - NVIDIA Data Center
    "V100": HardwareSpecs(
        name="NVIDIA Tesla V100",
        hardware_type=HardwareType.PROFESSIONAL_GPU,
        memory_gb=32.0,
        memory_bandwidth_gbps=900.0,
        memory_type="HBM2",
        compute_units=80,
        base_clock_mhz=1245,
        boost_clock_mhz=1380,
        fp16_tflops=31.4,
        fp32_tflops=15.7,
        int8_tops=125.6,
        power_watts=300,
        tdp_watts=300,
        performance_multiplier=1.2,
        efficiency_score=0.052,
        msrp_usd=8000,
        cost_per_hour_local=0.444,
        cost_per_hour_cloud=3.06,
        tensor_cores=True,
        nvlink=True,
        multi_gpu_scaling=0.9
    ),
    
    "A100_40GB": HardwareSpecs(
        name="NVIDIA A100 40GB",
        hardware_type=HardwareType.PROFESSIONAL_GPU,
        memory_gb=40.0,
        memory_bandwidth_gbps=1555.0,
        memory_type="HBM2e",
        compute_units=108,
        base_clock_mhz=765,
        boost_clock_mhz=1410,
        fp16_tflops=77.9,
        fp32_tflops=19.5,
        int8_tops=311.6,
        power_watts=400,
        tdp_watts=400,
        performance_multiplier=0.4,
        efficiency_score=0.049,
        msrp_usd=10000,
        cost_per_hour_local=0.556,
        cost_per_hour_cloud=4.10,
        tensor_cores=True,
        supports_int4=True,
        nvlink=True,
        multi_gpu_scaling=0.95
    ),
    
    "A100_80GB": HardwareSpecs(
        name="NVIDIA A100 80GB",
        hardware_type=HardwareType.PROFESSIONAL_GPU,
        memory_gb=80.0,
        memory_bandwidth_gbps=1935.0,
        memory_type="HBM2e",
        compute_units=108,
        base_clock_mhz=765,
        boost_clock_mhz=1410,
        fp16_tflops=77.9,
        fp32_tflops=19.5,
        int8_tops=311.6,
        power_watts=400,
        tdp_watts=400,
        performance_multiplier=0.4,
        efficiency_score=0.049,
        msrp_usd=15000,
        cost_per_hour_local=0.833,
        cost_per_hour_cloud=4.90,
        tensor_cores=True,
        supports_int4=True,
        nvlink=True,
        multi_gpu_scaling=0.95
    ),
    
    "H100": HardwareSpecs(
        name="NVIDIA H100",
        hardware_type=HardwareType.PROFESSIONAL_GPU,
        memory_gb=80.0,
        memory_bandwidth_gbps=3350.0,
        memory_type="HBM3",
        compute_units=132,
        base_clock_mhz=1095,
        boost_clock_mhz=1980,
        fp16_tflops=204.0,
        fp32_tflops=51.0,
        int8_tops=816.0,
        power_watts=700,
        tdp_watts=700,
        performance_multiplier=0.25,
        efficiency_score=0.073,
        msrp_usd=25000,
        cost_per_hour_local=1.389,
        cost_per_hour_cloud=8.00,
        tensor_cores=True,
        supports_int4=True,
        nvlink=True,
        multi_gpu_scaling=0.98
    ),
    
    # CPU Configurations
    "CPU_16GB": HardwareSpecs(
        name="CPU 16GB (Intel i7/AMD Ryzen 7)",
        hardware_type=HardwareType.CPU,
        memory_gb=16.0,
        memory_bandwidth_gbps=51.2,  # DDR4-3200
        memory_type="DDR4",
        compute_units=8,  # Cores
        base_clock_mhz=3000,
        boost_clock_mhz=4500,
        fp16_tflops=0.5,
        fp32_tflops=0.25,
        int8_tops=2.0,
        power_watts=65,
        tdp_watts=65,
        performance_multiplier=10.0,  # Much slower for LLM inference
        efficiency_score=0.004,
        msrp_usd=300,
        cost_per_hour_local=0.017,
        cost_per_hour_cloud=0.20,
        supports_fp16=False,
        supports_int8=True,
        tensor_cores=False
    ),
    
    "CPU_32GB": HardwareSpecs(
        name="CPU 32GB (Intel i9/AMD Ryzen 9)",
        hardware_type=HardwareType.CPU,
        memory_gb=32.0,
        memory_bandwidth_gbps=76.8,  # DDR4-4800
        memory_type="DDR4",
        compute_units=16,
        base_clock_mhz=3200,
        boost_clock_mhz=5000,
        fp16_tflops=0.8,
        fp32_tflops=0.4,
        int8_tops=3.2,
        power_watts=125,
        tdp_watts=125,
        performance_multiplier=8.0,
        efficiency_score=0.003,
        msrp_usd=500,
        cost_per_hour_local=0.028,
        cost_per_hour_cloud=0.35,
        supports_fp16=False,
        supports_int8=True,
        tensor_cores=False
    ),
    
    "CPU_64GB": HardwareSpecs(
        name="CPU 64GB (Intel Xeon/AMD EPYC)",
        hardware_type=HardwareType.CPU,
        memory_gb=64.0,
        memory_bandwidth_gbps=204.8,  # DDR5-6400
        memory_type="DDR5",
        compute_units=32,
        base_clock_mhz=2400,
        boost_clock_mhz=4000,
        fp16_tflops=1.5,
        fp32_tflops=0.75,
        int8_tops=6.0,
        power_watts=200,
        tdp_watts=200,
        performance_multiplier=6.0,
        efficiency_score=0.004,
        msrp_usd=1000,
        cost_per_hour_local=0.056,
        cost_per_hour_cloud=0.80,
        supports_fp16=False,
        supports_int8=True,
        tensor_cores=False
    ),
    
    # Apple Silicon
    "M1_Pro": HardwareSpecs(
        name="Apple M1 Pro",
        hardware_type=HardwareType.MOBILE,
        memory_gb=16.0,
        memory_bandwidth_gbps=200.0,
        memory_type="Unified",
        compute_units=16,  # GPU cores
        base_clock_mhz=3200,
        boost_clock_mhz=3200,
        fp16_tflops=5.2,
        fp32_tflops=2.6,
        int8_tops=10.4,
        power_watts=30,
        tdp_watts=30,
        performance_multiplier=2.0,
        efficiency_score=0.087,
        msrp_usd=1999,  # MacBook Pro price
        cost_per_hour_local=0.111,
        cost_per_hour_cloud=0.50,
        supports_int4=True,
        tensor_cores=False
    ),
    
    "M1_Max": HardwareSpecs(
        name="Apple M1 Max",
        hardware_type=HardwareType.MOBILE,
        memory_gb=32.0,
        memory_bandwidth_gbps=400.0,
        memory_type="Unified",
        compute_units=32,
        base_clock_mhz=3200,
        boost_clock_mhz=3200,
        fp16_tflops=10.4,
        fp32_tflops=5.2,
        int8_tops=20.8,
        power_watts=60,
        tdp_watts=60,
        performance_multiplier=1.5,
        efficiency_score=0.087,
        msrp_usd=3499,
        cost_per_hour_local=0.194,
        cost_per_hour_cloud=0.80,
        supports_int4=True,
        tensor_cores=False
    ),
    
    "M2_Ultra": HardwareSpecs(
        name="Apple M2 Ultra",
        hardware_type=HardwareType.MOBILE,
        memory_gb=128.0,
        memory_bandwidth_gbps=800.0,
        memory_type="Unified",
        compute_units=76,
        base_clock_mhz=3500,
        boost_clock_mhz=3500,
        fp16_tflops=27.2,
        fp32_tflops=13.6,
        int8_tops=54.4,
        power_watts=100,
        tdp_watts=100,
        performance_multiplier=0.8,
        efficiency_score=0.136,
        msrp_usd=6999,
        cost_per_hour_local=0.389,
        cost_per_hour_cloud=1.50,
        supports_int4=True,
        tensor_cores=False
    )
}


# Hardware categories
HARDWARE_CATEGORIES = {
    "consumer_gpu": ["GTX_1650", "RTX_3080", "RTX_3090", "RTX_4080", "RTX_4090"],
    "professional_gpu": ["V100", "A100_40GB", "A100_80GB", "H100"],
    "cpu": ["CPU_16GB", "CPU_32GB", "CPU_64GB"],
    "apple_silicon": ["M1_Pro", "M1_Max", "M2_Ultra"]
}


# Cloud instance mappings
CLOUD_INSTANCES = {
    "aws": {
        "GTX_1650": "g4dn.large",
        "RTX_3080": "g4dn.xlarge",
        "RTX_4090": "g5.xlarge",
        "V100": "p3.2xlarge",
        "A100_40GB": "p4d.xlarge",
        "A100_80GB": "p4d.2xlarge",
        "H100": "p5.xlarge"
    },
    "gcp": {
        "GTX_1650": "n1-standard-2-t4",
        "RTX_3080": "n1-standard-4-k80",
        "RTX_4090": "n1-standard-4-t4",
        "V100": "n1-standard-4-v100",
        "A100_40GB": "a2-highgpu-1g",
        "A100_80GB": "a2-highgpu-2g",
        "H100": "a3-highgpu-8g"
    },
    "azure": {
        "GTX_1650": "Standard_NC4as_T4_v3",
        "RTX_3080": "Standard_NC6s_v3",
        "RTX_4090": "Standard_NC4as_T4_v3",
        "V100": "Standard_NC6s_v3",
        "A100_40GB": "Standard_ND96asr_v4",
        "A100_80GB": "Standard_ND96amsr_A100_v4",
        "H100": "Standard_ND96isr_H100_v5"
    }
}


def get_hardware_specs(hardware_name: str) -> HardwareSpecs:
    """Get hardware specifications by name.
    
    Args:
        hardware_name: Name of the hardware
        
    Returns:
        HardwareSpecs object
        
    Raises:
        ValueError: If hardware is not found
    """
    if hardware_name not in HARDWARE_DATABASE:
        raise ValueError(f"Hardware '{hardware_name}' not found. Available hardware: {list(HARDWARE_DATABASE.keys())}")
    
    return HARDWARE_DATABASE[hardware_name]


def get_hardware_by_category(category: str) -> List[str]:
    """Get list of hardware in a category.
    
    Args:
        category: Category name
        
    Returns:
        List of hardware names
        
    Raises:
        ValueError: If category is not found
    """
    if category not in HARDWARE_CATEGORIES:
        raise ValueError(f"Category '{category}' not found. Available categories: {list(HARDWARE_CATEGORIES.keys())}")
    
    return HARDWARE_CATEGORIES[category]


def get_cloud_instance(hardware_name: str, provider: str = "aws") -> Optional[str]:
    """Get cloud instance name for hardware.
    
    Args:
        hardware_name: Name of the hardware
        provider: Cloud provider (aws, gcp, azure)
        
    Returns:
        Instance name or None if not available
    """
    if provider not in CLOUD_INSTANCES:
        return None
    
    return CLOUD_INSTANCES[provider].get(hardware_name)


def compare_hardware(hardware_names: List[str], metric: str = "fp16_tflops") -> List[tuple[str, float]]:
    """Compare hardware by a specific metric.
    
    Args:
        hardware_names: List of hardware names to compare
        metric: Metric to compare
        
    Returns:
        List of (hardware_name, metric_value) tuples sorted by metric
    """
    results = []
    
    for hardware_name in hardware_names:
        try:
            hardware_spec = get_hardware_specs(hardware_name)
            if hasattr(hardware_spec, metric):
                value = getattr(hardware_spec, metric)
                results.append((hardware_name, value))
        except ValueError:
            continue
    
    # Sort by metric value (descending for most metrics)
    reverse = metric not in ["performance_multiplier", "power_watts", "cost_per_hour_local", "cost_per_hour_cloud"]
    results.sort(key=lambda x: x[1], reverse=reverse)
    
    return results


def find_compatible_hardware(min_memory_gb: float, max_cost_per_hour: float = float('inf')) -> List[str]:
    """Find hardware that meets memory and cost requirements.
    
    Args:
        min_memory_gb: Minimum memory requirement in GB
        max_cost_per_hour: Maximum cost per hour
        
    Returns:
        List of compatible hardware names
    """
    compatible = []
    
    for name, spec in HARDWARE_DATABASE.items():
        if (spec.memory_gb >= min_memory_gb and 
            spec.cost_per_hour_local <= max_cost_per_hour):
            compatible.append(name)
    
    return compatible


def get_multi_gpu_scaling(hardware_name: str, num_gpus: int) -> float:
    """Calculate multi-GPU scaling efficiency.
    
    Args:
        hardware_name: Name of the hardware
        num_gpus: Number of GPUs
        
    Returns:
        Scaling efficiency factor
    """
    try:
        spec = get_hardware_specs(hardware_name)
        if num_gpus <= 1:
            return 1.0
        
        # Calculate scaling based on hardware capabilities
        base_scaling = spec.multi_gpu_scaling
        
        # Diminishing returns with more GPUs
        scaling_factor = base_scaling ** (num_gpus - 1)
        
        return scaling_factor
    except ValueError:
        return 0.8  # Default conservative scaling


def estimate_power_cost(hardware_name: str, hours: float, electricity_rate: float = 0.12) -> float:
    """Estimate power cost for running hardware.
    
    Args:
        hardware_name: Name of the hardware
        hours: Number of hours
        electricity_rate: Cost per kWh in USD
        
    Returns:
        Estimated power cost in USD
    """
    try:
        spec = get_hardware_specs(hardware_name)
        power_kwh = spec.power_watts / 1000
        cost = power_kwh * hours * electricity_rate
        return cost
    except ValueError:
        return 0.0


def get_all_hardware() -> List[str]:
    """Get list of all available hardware.
    
    Returns:
        List of hardware names
    """
    return list(HARDWARE_DATABASE.keys())


def get_hardware_recommendations(model_memory_gb: float, budget_per_hour: float = None) -> Dict[str, List[str]]:
    """Get hardware recommendations based on requirements.
    
    Args:
        model_memory_gb: Required memory for the model
        budget_per_hour: Optional budget constraint
        
    Returns:
        Dictionary with recommendations by category
    """
    recommendations = {
        "minimum": [],
        "recommended": [],
        "optimal": []
    }
    
    # Add 50% overhead for KV cache and activations
    required_memory = model_memory_gb * 1.5
    
    for name, spec in HARDWARE_DATABASE.items():
        if spec.memory_gb < required_memory:
            continue
            
        if budget_per_hour and spec.cost_per_hour_local > budget_per_hour:
            continue
        
        # Categorize based on memory headroom and performance
        memory_ratio = spec.memory_gb / required_memory
        
        if memory_ratio >= 3.0 and spec.performance_multiplier <= 0.5:
            recommendations["optimal"].append(name)
        elif memory_ratio >= 2.0 and spec.performance_multiplier <= 1.0:
            recommendations["recommended"].append(name)
        else:
            recommendations["minimum"].append(name)
    
    return recommendations