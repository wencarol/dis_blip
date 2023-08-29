#!/bin/bash
#SBATCH -J test
#SBATCH -N 4
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -p gpu-normal
#SBATCH --gres=gpu:4
#SBATCH -o test.out
#SBATCH -e test.err

export CUDA_VISIBLE_DEVICES=0,1,2,3

srun python -m torch.distributed.run --nproc_per_node=4 train_retrieval.py --config ./configs/retrieval_flickr_small6.yaml --output_dir output/retrieval_flickr
