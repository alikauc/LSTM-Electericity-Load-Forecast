#!/bin/bash
#SBATCH --job-name=lstm_forecast
#SBATCH --output=lstm_forecast_%j.log
#SBATCH --error=lstm_forecast_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1
#SBATCH --account=def-csimo_gpu

# Load necessary modules
module load python/3.11.5 scipy-stack

# Activate virtual environment
source /home/alika/projects/def-csimo/alika/LSTM-Electericity-Load-Forecast/env/bin/activate

# Print diagnostic info
echo "CUDA Available in PyTorch:"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# Execute training script
python /home/alika/projects/def-csimo/alika/LSTM-Electericity-Load-Forecast/train.py
