#!/usr/bin/env python3
"""
Setup script for LLM Inference Calculator

Installs the package and its dependencies.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-inference-calculator",
    version="1.0.0",
    author="LLM Calculator Team",
    author_email="team@llmcalculator.com",
    description="A comprehensive tool for estimating LLM inference costs, latency, and memory usage",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-inference-calculator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "ml": [
            "torch>=1.9.0",
            "transformers>=4.10.0",
            "accelerate>=0.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "llm-calc=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["*.py"],
        "examples": ["*.py"],
        "tests": ["*.py"],
        "research": ["*.md"],
        "scenarios": ["*.md"],
    },
    keywords="llm inference cost latency memory calculator ai ml",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/llm-inference-calculator/issues",
        "Source": "https://github.com/yourusername/llm-inference-calculator",
        "Documentation": "https://llm-inference-calculator.readthedocs.io/",
    },
)