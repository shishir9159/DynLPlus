#!/usr/bin/env bash
# Example SLURM job. Partition/account/module names are site-specific: edit them.
#SBATCH --job-name=dynlp-plus
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm-%j.out
# #SBATCH --partition=gpu
# #SBATCH --account=YOUR_ALLOCATION

set -euo pipefail
# module load cuda/12.4 python/3.11   # if your site needs it
cd "$SLURM_SUBMIT_DIR"
nvidia-smi
STAGE=${STAGE:-all}
bash scripts/run_h100.sh "$STAGE"
