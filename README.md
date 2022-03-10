# Confident Landmark Ensembling

This code can be used to reproduce the experiments performed in our paper 'Confidence is Key - A Weighted Ensembling of CNNs for Accurate Landmark Detection'.

## Requirements

- Python 3 (code has been tested on Python 3.7)
- CUDA and cuDNN (tested with Cuda 11.3)
- Our experiments used a n NVIDIA Tesla V100 GPU with 32GB of memory (of which 24GB is used).
- Python packages listed in the requirements.txt including PyTorch 1.10.0

## Getting Started

1. Go to your chosen directory, clone this repo then enter it:
```
git clone https://github.com/jfm15/ConfidentLandmarkEnsembling.git
cd ConfidentLandmarkEnsembling/
```

2. Install required packages. In this guide we create our own virtual environment:

```
python3 -m venv {virtual_environment_name}
source {virtual_environment_name}/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset Preparation

1. Download the cephalometric dataset from the link: http://www-o.ntust.edu.tw/~cweiwang/ISBI2015/challenge1/

2. Extract the folders 'RawImage' and 'AnnotationsByMD' into a directory of your choosing such that the file structure looks like this:

````bash
{cephalometric_data_directory}
├── AnnotationsByMD
│   ├── 400_junior
│   │   ├── 001.txt
│   │   ├── 002.txt
│   │   ├── ...
│   │   └── 400.txt
│   │
│   └── 400_senior
│   │   ├── 001.txt
│   │   ├── 002.txt
│   │   ├── ...
│   │   └── 400.txt
│   │
└── RawImage
    ├── TrainingData
    │   ├── 001.bmp
    │   ├── 002.bmp
    │   ├── ...
    │   └── 150.bmp
    │
    └── Test1Data
    │   ├── 151.bmp
    │   ├── 152.bmp
    │   ├── ...
    │   └── 300.bmp
    │
    └── Test2Data
        ├── 301.bmp
        ├── 302.bmp
        ├── ...
        └── 400.bmp
````

Note that if you publish work using this dataset you must cite:
````
@article{wang2016benchmark,
  title={A benchmark for comparison of dental radiography analysis algorithms},
  author={Wang, Ching-Wei and Huang, Cheng-Ta and Lee, Jia-Hong and Li, Chung-Hsing and Chang, Sheng-Wei and Siao, Ming-Jhih and Lai, Tat-Ming and Ibragimov, Bulat and Vrtovec, Toma{\v{z}} and Ronneberger, Olaf and others},
  journal={Medical image analysis},
  volume={31},
  pages={63--76},
  year={2016},
  publisher={Elsevier}
}
````

### Running The Code

You can either train the model yourself or download one of our pretrained models.

#### 1. Train a model

1.1 Train a model using the following command. This script resizes images in your training set directory 
and saves them in ConfidentLandmarkEnsembling/cache. This script will perform the experiment 3 times. Each experiment will 
train the ensemble from scratch and after 10 epochs it will save the 3 base estimators in {output_path}/ceph_sup_150/run:{X}_models/ where {X} is replaced by 0, 1 or 2 depending on which repetition the
experiment is.

```
python train.py --cfg experiments/cephalometric/ceph_sup_150.yaml --training_images {cephalometric_data_directory}/RawImage/TrainingData/ \
 --validation_images /data/coml-oxmedis/shug6372/data/CephalometricData/RawImage/Test1Data/ /data/coml-oxmedis/shug6372/data/CephalometricData/RawImage/Test2Data/ \
 --annotations {cephalometric_data_directory}/AnnotationsByMD/ --output_path {output_path}

(To be completed on March 10th)