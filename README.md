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

#### 1. Cephalometric Dataset

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

#### 2. 2D Hand Dataset

The images for the 2D Hand experiments can be downloaded here: https://ipilab.usc.edu/research/baaweb/

The corresponding annotations were obtained from the following github repository: https://github.com/christianpayer/MedicalDataAugmentationTool-HeatmapRegression. We would like to thank Payer et al for making these annotations public and if you use this data please cite them using:

````
@article{Payer2019a,
  title   = {Integrating Spatial Configuration into Heatmap Regression Based {CNNs} for Landmark Localization},
  author  = {Payer, Christian and {\v{S}}tern, Darko and Bischof, Horst and Urschler, Martin},
  journal = {Medical Image Analysis},
  volume  = {54},
  year    = {2019},
  month   = {may},
  pages   = {207--219},
  doi     = {10.1016/j.media.2019.03.007},
}
````

We use some additional scripts to format the hand dataset into the following file structure:

````bash
{hand_data_directory}
├── annotations
│   ├── txt
│   │   ├── 3142.txt
│   │   ├── 3143.txt
│   │   ├── ...
│   │   └── 7293.txt
│   │
└── set_1
    ├── training
    │   ├── 3142.jpg
    │   ├── 3143.jpg
    │   ├── ...
    │   └── 7293.jpg
    │
    └── validation
        ├── 3147.jpg
        ├── 3151.jpg
        ├── ...
        └── 7288.jpg

````

Where the images are placed into either the training or validation directory based on the predefined set 1 configuration. 
The associated txt files contain the ground truth annotations.

### Running The Code

#### 1. Train a model - Cephalometric

Train a cephalometric model using the following command. This script resizes images in your training set directory 
and saves them in ConfidentLandmarkEnsembling/cache. This script will perform the experiment 3 times. Each experiment will 
train the ensemble from scratch and after 10 epochs it will save the 3 base estimators in {output_path}/ceph_sup_150/run:{X}_models/ where {X} is replaced by 0, 1 or 2 depending on which repetition the
experiment is.

```
python train.py --cfg experiments/cephalometric/ceph_sup_150.yaml --training_images {cephalometric_data_directory}/RawImage/TrainingData/ \
 --validation_images {cephalometric_data_directory}/RawImage/Test1Data/ {cephalometric_data_directory}/RawImage/Test2Data/ \
 --annotations {cephalometric_data_directory}/AnnotationsByMD/ --output_path {output_path}
```

#### 2. Train a model - 2D Hand Dataset

In a similar fashion to the cephalometric this script will resize images and place them in ConfidentLandmarkEnsembling/cache. It will
perform the experiment 3 times and will save a base estimator in {output_path}/hands_sup_100%/run:{X}_models/ where {X} is replaced by 0, 1 or 2 depending on which repetition the
experiment is.


```
python train.py --cfg --cfg experiments/hands/hands_sup_100%.yaml --training_images {hand_data_directory}/set_1/training/ \
 --validation_images {hand_data_directory}/set_1/validation/ \
 --annotations {hand_data_directory}/annotations --output_path {output_path}
```

#### 3. Validation

When running these scripts, at the end of each training loop, it will validate the base estimators and aggregation methods on the validation set specified in the command line. These results will be inline with the results reported in the paper. Here is an example:

```
-----------Validating over /data/coml-oxmedis/shug6372/data/CephalometricData/RawImage/Test1Data/-----------

-----------Statistics for /data/coml-oxmedis/shug6372/data/CephalometricData/RawImage/Test1Data/-----------
Average radial error per base model: 1.120mm 1.099mm 1.096mm
Average radial error for mean average aggregation: 1.062mm
Average radial error for confidence weighted aggregation: 1.058mm
The Successful Detection Rate (SDR) for confidence weighted aggregation for thresholds 2.0mm, 2.5mm, 3.0mm, 4.0mm respectively is 88.211%, 92.526%, 95.193%, 98.211%

-----------Validating over /data/coml-oxmedis/shug6372/data/CephalometricData/RawImage/Test2Data/-----------

-----------Statistics for /data/coml-oxmedis/shug6372/data/CephalometricData/RawImage/Test2Data/-----------
Average radial error per base model: 1.418mm 1.384mm 1.385mm
Average radial error for mean average aggregation: 1.361mm
Average radial error for confidence weighted aggregation: 1.351mm
The Successful Detection Rate (SDR) for confidence weighted aggregation for thresholds 2.0mm, 2.5mm, 3.0mm, 4.0mm respectively is 77.684%, 85.158%, 89.105%, 94.579%

-----------Combined Statistics-----------
Average radial error per base model: 1.229mm 1.213mm 1.212mm
Average radial error for mean average aggregation: 1.1816mm
Average radial error for confidence weighted aggregation: 1.1752mm
The Successful Detection Rate (SDR) for confidence weighted aggregation for thresholds 2.0mm, 2.5mm, 3.0mm, 4.0mm respectively is 84.000%, 89.579%, 92.758%, 96.758%
```