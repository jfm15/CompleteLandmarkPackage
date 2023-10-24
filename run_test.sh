#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --clusters=arc
#SBATCH --partition=medium

module load Anaconda3/2020.11
source activate /data/coml-oxmedis/kebl7678/conda_envs/chm_env

#run python code
python tools/test.py --cfg experiments/ultra_hip/ddh.yaml --images /data/coml-oxmedis/datasets-in-use/ultrasound-hip-baby-land-seg/images/img --annotations /data/coml-oxmedis/datasets-in-use/ultrasound-hip-baby-land-seg/annotations/txt --pretrained_model_directory './ddh/training/run:0_models' --partition /data/coml-oxmedis/datasets-in-use/ultrasound-hip-baby-land-seg/partitions/partition_abhi_89_47_49.json