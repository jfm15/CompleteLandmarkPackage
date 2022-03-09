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
git clone https://github.com/jfm15/ContourHuggingHeatmaps.git
cd ContourHuggingHeatmaps/
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
and saves them in ContourHuggingHeatmaps/cache. After 10 epochs it will save the model at
ContourHuggingHeatmaps/output/cephalometric/cephalometric_model.pth.

```
python train.py --cfg experiments/cephalometric.yaml --training_images {cephalometric_data_directory}/RawImage/TrainingData/ \
 --annotations {cephalometric_data_directory}/AnnotationsByMD/
```

1.2 Perform temperature scaling on the model saved in the previous step using the following command. 
The model with the best Estimated Calibration Error (ECE) score will be saved at ContourHuggingHeatmaps/output/cephalometric/cephalometric_scaled_model.pth.

```
python temperature_scaling.py --cfg experiments/cephalometric.yaml --fine_tuning_images {cephalometric_data_directory}/RawImage/Test1Data/ \
 --annotations {cephalometric_data_directory}/AnnotationsByMD/ --pretrained_model output/cephalometric/cephalometric_model.pth
```

#### 2. Download a model

2.1 If you would like, instead of training a model you can download our pretrained models at the following link: https://app.box.com/s/4qz3tthh7q6xajtaasj4fp9iaw86mmyx

#### 3. Testing

3.1 Test the models using the following commands where {model_path} is the path to the model you have trained or downloaded. 
You can either test the basic model or the temperature scaled model.

```
python test.py --cfg experiments/cephalometric.yaml --testing_images {cephalometric_data_directory}/RawImage/{Test1Data or Test2Data}/
--annotations {cephalometric_data_directory}/AnnotationsByMD  --pretrained_model {model_path}
```