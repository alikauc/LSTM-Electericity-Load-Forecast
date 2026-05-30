#!/bin/bash
#SBATCH --job-name=arima_sweep
#SBATCH --output=logs/arima_sweep_%A_%a.log
#SBATCH --error=logs/arima_sweep_%A_%a.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --account=def-csimo
#SBATCH --array=0-50

# Ensure the logs directory exists
mkdir -p logs

# Load required cluster modules
module load python/3.11.5 scipy-stack

# Activate python virtual environment
source /home/alika/projects/def-csimo/alika/LSTM-Electericity-Load-Forecast/env/bin/activate

# Execute a single parameter combination based on the SLURM Array Task ID.
# Using a daily stride (24) on the validation split yields fast and highly accurate parameter sweep scores.
python /home/alika/projects/def-csimo/alika/LSTM-Electericity-Load-Forecast/test_arima_only.py \
    --mode evaluate \
    --array-index $SLURM_ARRAY_TASK_ID \
    --stride 24 \
    --split val \
    --n-jobs 2
