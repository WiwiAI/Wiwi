#!/bin/bash
# Wiwi4.0 Launch Script

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wiwi

# Set LD_LIBRARY_PATH for cuDNN (required for faster-whisper/CTranslate2)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

# CUDA architecture for DeepSpeed compilation (GTX 1650 SUPER = 7.5, CMP 50HX = 8.6)
# This speeds up first-time DeepSpeed kernel compilation
export TORCH_CUDA_ARCH_LIST="7.5;8.6"

# Run Wiwi --log-level DEBUG 
python -m wiwi "$@"
