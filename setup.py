#!/usr/bin/env python3
"""
Setup script for the lotteries project.
This script will create necessary directories and install required packages.
"""

import os
import subprocess
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories for the project."""
    directories = [
        "data",
        "output",
        "Edreams_output",
        "lotto_lab_out"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def install_requirements():
    """Install required packages from requirements.txt."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Successfully installed all requirements")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        print("Please install the packages manually using: pip install -r requirements.txt")

def create_sample_data():
    """Create sample data files for testing."""
    try:
        import numpy as np
        import pandas as pd
        
        # Create sample data for grok.py files
        np.random.seed(42)
        sequence_a = pd.Series(np.random.randn(1000).cumsum())
        sequence_b = pd.Series(0.8 * sequence_a + 0.2 * np.random.randn(1000))
        
        sequence_a.to_csv("data/g1.csv", index=False, header=False)
        sequence_b.to_csv("data/poi.csv", index=False, header=False)
        
        print("✓ Created sample data files in data/ directory")
        
    except ImportError:
        print("⚠ Could not create sample data files (numpy/pandas not available)")
        print("  This is normal if packages aren't installed yet")

def main():
    """Main setup function."""
    print("Setting up lotteries project...")
    print("=" * 50)
    
    create_directories()
    print()
    
    if Path("requirements.txt").exists():
        print("Installing Python packages...")
        install_requirements()
        print()
    
    print("Creating sample data files...")
    create_sample_data()
    print()
    
    print("Setup complete!")
    print("\nNext steps:")
    print("1. Adjust file paths in the Python scripts to match your data")
    print("2. Install any missing dependencies: pip install -r requirements.txt")
    print("3. Run your lottery analysis scripts")

if __name__ == "__main__":
    main()