# Slurm Quick Commands
squeue -u <username>

# Slurm Utilities
sbatch <job-script.sh>
squeue -u <username>
scancel <job-id>

# Accessing Server
ssh <username>@<login-host>

# Optional SSH Alias (example)
ssh <alias> <username>@<cluster-host>  # then use: ssh <alias>

# Allocate Interactive Compute Node
salloc -N1 --nodelist=<node-name> --partition=general
srun --pty /bin/bash

# Scratch Storage Paths
/scratch/path_a
/scratch/path_b

# Private Tokens and Secrets
Store private values only in `server.local.md` (do not commit).
Example:
dotloop: <your-private-token>

# Example Slurm Batch Script
#!/bin/bash
#SBATCH --job-name=extract_features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=extract_features_%j.log
#SBATCH --error=extract_features_%j.err
#SBATCH --partition=general

conda activate <env-name>
bash train.bash

# Another Slurm Script Example
#!/bin/bash

#SBATCH --job-name=train_grpo_cot
#SBATCH --partition=general
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --output=train_grpo_cot_%j.log
#SBATCH --error=train_grpo_cot_%j.err
#SBATCH --nodelist=<node-name>

source ~/.bashrc
conda activate <env-name>

CUDA_VISIBLE_DEVICES=0,1 python train_grpo.py --mode cot --cold_start_model_path <model-path>
