# Environment Setup Guide

This guide will help you set up a clean, isolated environment for the LLM Inference Calculator project.

## 🐍 Python Virtual Environment Setup

### Option 1: Using venv (Recommended)

```bash
# Navigate to project directory
cd d:\misogiai\week-3\day-3\w3d3a1

# Create virtual environment
python -m venv llm_calc_env

# Activate virtual environment
# On Windows Command Prompt:
llm_calc_env\Scripts\activate

# On Windows PowerShell:
llm_calc_env\Scripts\Activate.ps1

# On Windows Git Bash/WSL/Unix-like shells:
source llm_calc_env/Scripts/activate

# On macOS/Linux:
source llm_calc_env/bin/activate

# Verify activation (should show virtual env path)
which python
```

### Option 2: Using conda

```bash
# Create conda environment
conda create -n llm_calc python=3.9

# Activate environment
conda activate llm_calc

# Navigate to project directory
cd d:\misogiai\week-3\day-3\w3d3a1
```

## 📦 Install Dependencies

### Basic Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Full Installation with All Features

```bash
# Install with all optional dependencies
pip install -e ".[dev,viz,ml]"
```

### Minimal Installation (Core Only)

```bash
# Install only essential dependencies
pip install numpy pandas click rich
```

## 🔧 Development Setup

For development work, install additional tools:

```bash
# Install development dependencies
pip install -e ".[dev]"

# This includes:
# - pytest (testing)
# - black (code formatting)
# - flake8 (linting)
# - mypy (type checking)
```

## 🧪 Verify Installation

### Quick Test

```bash
# Test basic functionality
python quick_start.py

# Test command-line interface
python app.py --help

# Run a simple calculation
python app.py --model llama-2-7b --tokens 100 --hardware rtx-4090
```

### Run Test Suite

```bash
# Run all tests
python run_tests.py

# Or using pytest directly
pytest tests/ -v
```

## 📊 Optional: Visualization Dependencies

For plotting and visualization features:

```bash
# Install visualization libraries
pip install ".[viz]"

# This includes:
# - matplotlib
# - seaborn
# - plotly
```

## 🤖 Optional: Machine Learning Dependencies

For advanced ML features (model loading, etc.):

```bash
# Install ML libraries (large download)
pip install ".[ml]"

# This includes:
# - torch
# - transformers
# - accelerate
```

## 🔍 Troubleshooting

### Common Issues

1. **Permission Denied Errors**
   
   **Error**: `[Errno 13] Permission denied` when creating virtual environments
   
   **Solution 1: Run as Administrator (Windows)**
   1. Right-click Command Prompt or PowerShell
   2. Select "Run as administrator"
   3. Navigate to your project directory
   4. Run the virtual environment creation command
   
   **Solution 2: Use User Directory (Recommended)**
   ```cmd
   # Create venv in a user-accessible location
   python -m venv %USERPROFILE%\llm_calc_env
   %USERPROFILE%\llm_calc_env\Scripts\activate
   cd /d "d:\misogiai\week-3\day-3\w3d3a1"
   pip install -r requirements.txt
   ```
   
   **Solution 3: Alternative Virtual Environment Tools**
   ```cmd
   # Using conda (if installed)
   conda create -n llm_calc_env python=3.8
   conda activate llm_calc_env
   pip install -r requirements.txt
   
   # Using virtualenv
   pip install virtualenv
   virtualenv llm_calc_env
   llm_calc_env\Scripts\activate
   ```

2. **Virtual Environment Activation Errors**
   
   **Error**: `bash: llm_calc_envScriptsactivate: command not found`
   
   **Solution**: You're using a Unix-like shell (Git Bash, WSL, etc.). Use the correct activation command:
   ```bash
   # For Git Bash/WSL/Unix shells on Windows:
   source llm_calc_env/Scripts/activate
   
   # Check which shell you're using:
   echo $0
   ```
   
   **Shell-specific activation commands:**
   - **Command Prompt**: `llm_calc_env\Scripts\activate`
   - **PowerShell**: `llm_calc_env\Scripts\Activate.ps1`
   - **Git Bash/WSL**: `source llm_calc_env/Scripts/activate`
   - **macOS/Linux**: `source llm_calc_env/bin/activate`

2. **Import Errors**
   ```bash
   # Make sure you're in the right directory
   cd d:\misogiai\week-3\day-3\w3d3a1
   
   # Reinstall in development mode
   pip install -e .
   ```

3. **Missing Dependencies**
   ```bash
   # Update pip first
   pip install --upgrade pip
   
   # Reinstall requirements
   pip install -r requirements.txt
   ```

4. **Virtual Environment Issues**
   ```bash
   # Deactivate and recreate environment
   deactivate
   rm -rf llm_calc_env  # or rmdir /s llm_calc_env on Windows
   python -m venv llm_calc_env
   # Use appropriate activation command for your shell
   ```

### Version Compatibility

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, Linux
- **Memory**: At least 4GB RAM recommended
- **Storage**: ~500MB for full installation with ML dependencies

## 🚀 Quick Start Commands

Once environment is set up:

```bash
# Interactive guided tour
python quick_start.py

# Basic calculation
python app.py --model llama-2-7b --tokens 100 --hardware rtx-4090

# Compare models
python app.py --compare --models llama-2-7b llama-2-13b

# Get hardware recommendations
python app.py --recommend-hardware --model llama-2-13b

# Interactive mode
python app.py --interactive

# Analyze scenarios
python app.py --scenario chatbot
```

## 📝 Environment Variables (Optional)

You can set these environment variables for customization:

```bash
# Windows
set LLM_CALC_DEBUG=1
set LLM_CALC_CACHE_DIR=./cache

# macOS/Linux
export LLM_CALC_DEBUG=1
export LLM_CALC_CACHE_DIR=./cache
```

## 🔄 Updating the Project

To update dependencies:

```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Reinstall the project
pip install -e . --force-reinstall
```

## 🧹 Cleanup

To remove the environment:

```bash
# Deactivate virtual environment
deactivate

# Remove environment directory
# Windows:
rmdir /s llm_calc_env

# macOS/Linux:
# rm -rf llm_calc_env

# Or for conda:
# conda env remove -n llm_calc
```

---

**Note**: Always activate your virtual environment before working with the project to ensure dependency isolation and avoid conflicts with other Python projects.