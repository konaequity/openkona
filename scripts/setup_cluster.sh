#!/bin/bash
# ============================================================
#  KONASH OAPL Training — Together AI Cluster Setup
# ============================================================
#  Run this after SSH-ing into your Together AI Slurm cluster:
#
#    ssh <cluster-ssh-command-from-ui>
#    srun --gpus=2 --pty bash
#    bash setup_cluster.sh
#
#  Prerequisites:
#    - Together AI account with GPU cluster (2x H100 SXM)
#    - SSH key added at api.together.ai/settings/ssh-key
#    - TOGETHER_API_KEY set (for rollout generation via API)
# ============================================================

set -euo pipefail

echo "============================================================"
echo "  KONASH Cluster Setup"
echo "============================================================"

# 1. Check GPUs
echo ""
echo "  Checking GPUs..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "  Found $GPU_COUNT GPU(s)"

if [ "$GPU_COUNT" -lt 2 ]; then
    echo "  WARNING: Expected 2 GPUs, found $GPU_COUNT"
fi

# 2. Install dependencies
echo ""
echo "  Installing dependencies..."
pip install --quiet --upgrade pip

# Unsloth (includes torch, transformers, peft, trl, vllm)
pip install --quiet unsloth

# KONASH
if [ -d "/mnt/shared/konash" ]; then
    echo "  Using repo from shared storage: /mnt/shared/konash"
    cd /mnt/shared/konash
else
    echo "  Cloning repo..."
    git clone https://github.com/your-org/konash.git /mnt/shared/konash
    cd /mnt/shared/konash
fi

pip install --quiet -e ".[train]"

# 3. Verify imports
echo ""
echo "  Verifying imports..."
python3 -c "
from unsloth import FastLanguageModel
from konash.training.unsloth_engine import UnslothEngine
from konash.training.oapl import OAPLTrainer
print('  All imports OK')
"

# 4. Set environment
export UNSLOTH_VLLM_STANDBY=1

echo ""
echo "============================================================"
echo "  Setup complete! Run training with:"
echo ""
echo "  # From pre-generated rollouts:"
echo "  python scripts/train_oapl_unsloth.py \\"
echo "      --rollouts glm_test_results/stage3_results.json \\"
echo "      --output /mnt/shared/checkpoints/iter1"
echo ""
echo "  # Full pipeline (synthesis + rollouts + OAPL):"
echo "  python scripts/train_oapl_unsloth.py \\"
echo "      --corpus /mnt/shared/corpus \\"
echo "      --iterations 2 \\"
echo "      --output /mnt/shared/checkpoints"
echo "============================================================"
