#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00

module load Anaconda3/2020.11
source activate /data/coml-oxmedis/kebl7678/conda_envs/chm_env

#run python code
#python tools/train.py --cfg experiments/ultra_hip/ddh.yaml --images /data/coml-oxmedis/datasets-in-use/ultrasound-hip-baby-land-seg/images/img --annotations /data/coml-oxmedis/datasets-in-use/ultrasound-hip-baby-land-seg/annotations/txt --partition '.partition_0.15_0.15_0.7_0.00000.json' --output_path './output/oai'
python tools/train.py --cfg experiments/oai/oai.yaml --images /data/coml-oxmedis/datasets-in-use/oai/simon_annotations/imgs --annotations /data/coml-oxmedis/datasets-in-use/oai/simon_annotations/txt --partition '/data/coml-oxmedis/datasets-in-use/oai/simon_annotations/partition_0.7_0.15_0.15_0.00000.json' --output_path './output/oai'
