"""
env_check.py — Simple environment verification script.

Run this script to verify that your Python environment is correctly set up
and has all the required dependencies installed.
"""

import sys
import importlib

def check_environment():
    """Checks Python version and required dependencies."""
    print("=" * 50)
    print("🔍 Environment Verification")
    print("=" * 50)
    
    # 1. Check Python version
    print(f"Python Version: {sys.version.split(' ')[0]}")
    if sys.version_info < (3, 9):
        print("⚠️ Warning: Python 3.9+ is recommended.")
    else:
        print("✅ Python version is compatible.")
    
    print("-" * 50)
    
    # 2. Check required packages
    required_packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "sklearn",
        "sentence-transformers": "sentence_transformers",
        "pyyaml": "yaml",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "jupyter": "jupyter",
        "tqdm": "tqdm"
    }
    
    missing_packages = []
    
    for display_name, import_name in required_packages.items():
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"✅ {display_name:<22} installed (version: {version})")
        except ImportError:
            print(f"❌ {display_name:<22} MISSING")
            missing_packages.append(display_name)
            
    print("-" * 50)
    
    # 3. Final verdict
    if missing_packages:
        print("🚨 Environment check FAILED.")
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
    else:
        print("🎉 Environment check PASSED! You are ready to go.")
    print("=" * 50)

if __name__ == "__main__":
    check_environment()
