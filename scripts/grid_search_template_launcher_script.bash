#!/bin/bash

######################################## BSUB Headers ###########################################
# Begin LSF Directives
#BSUB -P BIF135-TWO
#BSUB -W 02:00
#BSUB -nnodes 1
#BSUB -q batch
#BSUB -alloc_flags "gpumps"
#BSUB -J train_test_gcpnet
#BSUB -o job%J.out
#BSUB -e job%J.out
#################################################################################################

# project ID and directory structure(s)
export PROJID=bif135
export PROJDIR=/gpfs/alpine/scratch/"$USER"/"$PROJID"/Repositories/Lab_Repositories/GCPNet

# remote Conda environment
module purge
module load open-ce/1.5.2-py38-0 gcc/9.1.0
conda activate "$PROJDIR"/gcpnet/

# set PyTorch Lightning optimization flags for NCCL
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2

# establish reasonable arguments for OpenMP
export OMP_PLACES=threads

# configure WandB logger for local configuration storage and proxy access on compute nodes
export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_CONFIG_DIR=/gpfs/alpine/scratch/"$USER"/"$PROJID"/ # For local reading and writing of WandB files
export WANDB_CACHE_DIR=/gpfs/alpine/scratch/"$USER"/"$PROJID"/  # For logging checkpoints as artifacts
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,.ccs.ornl.gov,.ncrc.gov'
export LC_ALL=en_US.utf8
export WANDB_RESUME=allow

# configure PyTorch's extensions directory
mkdir -p /gpfs/alpine/scratch/"$USER"/"$PROJID"/Extensions
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/scratch/"$USER"/"$PROJID"/Extensions/torch_extensions/
mkdir -p "$TORCH_EXTENSIONS_DIR"

# move into source directory
cd "$PROJDIR" || exit

# execute script
