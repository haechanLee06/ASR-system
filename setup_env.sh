#!/bin/bash
# ASR System Environment Setup Script
# Creates conda environment 'asr_sys' with all dependencies

set -e

ENV_NAME="asr_sys"
PYTHON_VERSION="3.10"

echo "=========================================="
echo "ASR System Environment Setup"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Remove existing environment if exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[INFO] Removing existing ${ENV_NAME} environment..."
    conda env remove -n ${ENV_NAME} -y
fi

echo "[INFO] Creating conda environment: ${ENV_NAME} with Python ${PYTHON_VERSION}"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

echo "[INFO] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo "[INFO] Installing PyTorch with CUDA support..."
# Install PyTorch (adjust cuda version if needed)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "[INFO] Installing core dependencies..."
pip install \
    Flask==2.3.3 \
    Flask-SQLAlchemy==3.1.1 \
    SQLAlchemy==2.0.36 \
    requests==2.31.0 \
    flask-jwt-extended==4.6.0 \
    flask-cors==6.0.1

echo "[INFO] Installing FunASR and ModelScope..."
pip install funasr modelscope addict

echo "[INFO] Installing 3D-Speaker dependencies..."
pip install \
    tqdm>=4.42.0 \
    scipy>=1.7.0 \
    "numpy>=1.20.0,<2.0" \
    scikit-learn==1.0.2 \
    soundfile \
    kaldiio \
    pyyaml \
    matplotlib \
    pandas \
    openpyxl

echo "[INFO] Installing pyannote.audio (for overlap detection, optional)..."
pip install pyannote.audio || echo "[WARN] pyannote.audio installation failed, overlap detection will be disabled"

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "To start the server, run:"
echo "    python run.py"
echo ""
