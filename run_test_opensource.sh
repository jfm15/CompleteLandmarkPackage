#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --partition=medium

module load Anaconda3/2020.11
source activate /data/coml-oxmedis/kebl7678/conda_envs/chm_env

#run python code
python tools/test_opensource.py --cfg experiments/ultra_hip/ddh.yaml --images /data/coml-oxmedis/datasets-in-use/ultrasound-opensource/all_imgs_standardsize --annotations None --pretrained_model_directory './ddh/training/run:0_models' --partition /data/coml-oxmedis/datasets-in-use/ultrasound-opensource/partitions/partition_opensource.json

