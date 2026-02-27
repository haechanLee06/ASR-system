#!/usr/bin/env python
"""
安装3D-Speaker所需依赖
"""

import subprocess
import sys
import os

def install_packages():
    """安装所需的包"""
    packages = [
        "datasets",
        "pyannote.audio>=3.0",
        "modelscope",
        "funasr",
        "torch",
        "torchaudio",
        "numpy",
        "scipy",
        "tqdm",
        "soundfile",
        "huggingface_hub",
        "pyyaml"
    ]
    
    print("Installing required packages...")
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✓ {package} installed successfully")
            else:
                print(f"✗ Failed to install {package}")
                print(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⚠ Installation of {package} timed out")
        except Exception as e:
            print(f"✗ Error installing {package}: {str(e)}")
    
    print("\nInstallation completed. Please restart the application.")

def check_package_installed(package_name):
    """检查包是否已安装"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def check_requirements():
    """检查基本依赖"""
    required = ['torch', 'torchaudio', 'modelscope', 'funasr']
    missing = []
    
    for pkg in required:
        if not check_package_installed(pkg):
            missing.append(pkg)
    
    if missing:
        print(f"Missing packages: {missing}")
        return False
    else:
        print("All required packages are installed")
        return True

if __name__ == "__main__":
    print("3D-Speaker Dependency Installer")
    print("="*50)
    
    # 检查当前环境
    print("Checking current environment...")
    if check_requirements():
        print("Environment seems OK, skipping installation")
    else:
        print("Some packages are missing, proceeding with installation...")
        install_packages()