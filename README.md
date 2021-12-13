# CLTT
## Contrastive Learning Through Time

<p align="center">
  <img src="https://github.com/trieschlab/cltt/blob/release/img/header.png" width="500">

CLTT stands for Contrastive Learning Through Time. It is the codebase used for the SVRHM NeurIPS workshop publication "Contrastive Learning Through Time" [1]. 
If you make use of this code please cite as follows:
 

[1] **F. Schneider, X. Xu, M. R. Ernst, Z. Yu, and J. Triesch. Contrastive learning through time. In SVRHM 2021 Workshop @ NeurIPS, 2021.**


## Getting started with the repository

* Clone the repository from here
* Make sure you have all the dependencies installed
* Download the corresponding TDW, COIL100 or Miyashita dataset files
* Configure the config.py file
* Start an experiment on your local machine with python3 main/train.py

### Prerequisites

* [numpy](http://www.numpy.org/)
* [pytorch](https://www.pytorch.org/)
* [matplotlib](https://matplotlib.org/)
* [tensorboard](https://tensorflow.org/)
* [pandas](https://pandas.pydata.org)
* [tqdm](https://pypi.org/project/tqdm/)
* [pacmap](https://pypi.org/project/pacmap/)

### Datasets

#### Miyashita
The generator script for the Miyashita-style dataset can be found at [github.com/mrernst/miyashita_fractals](https://github.com/mrernst/miyashita_fractals). Run the script with options --imgsize 64 --stimuli 100 to get the 100 fractals used for this work.

#### TDW
The ThreeDWorld (TDW) dataset will be available from Zenodo at []{}

#### COIL-100
COIL-100 is publicly available at [https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php](https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php)

### Directory structure

```bash
.
├── codetemplate                       # Template for collaboration purposes
├── config.py                          # Configuration for experiment parameters
├── data                          
│   ├── coil100_128x128                # COIL100 dataset
│   ├── fractals100_64x64              # Miyashita fractals
│   └── spherical_photoreal_64x64_DoF  # ThreeDWorld dataset
├── img
│   └── header.png  				   # Image that displays in README.md
├── LICENSE                            # MIT License
├── main
│   ├── generic.sbatch
│   └── train.py             		                 		    
├── README.md                         # ReadMe File
├── requirements.txt                  # conda/pip requirements
├── utils
│   ├── augmentations.py           	  # augmentations for standard SimCLR
│   ├── datasets.py             	  # dataloading and sampling
│   ├── evaluation.py             	  # evaluation methods and analytics
│   ├── general.py					  # general utilities
│   ├── losses.py					  # definition of loss functions
└── └── networks.py					  # network definition, e.g. ResNet

```

### Installation guide

#### Forking the repository

Fork a copy of this repository onto your own GitHub account and `clone` your fork of the repository into your computer, inside your favorite folder, using:

`git clone "PATH_TO_FORKED_REPOSITORY"`

#### Setting up the environment

Install [Python 3.9](https://www.python.org/downloads/release/python-395/) and the [conda package manager](https://conda.io/miniconda.html) (use miniconda). Navigate to the project directory inside a terminal and create a virtual environment (replace <ENVIRONMENT_NAME>, for example, with `cltt`) and install the [required packages](requirements.txt):

`conda create -n <ENVIRONMENT_NAME> --file requirements.txt python=3.9`

Activate the virtual environment:

`source activate <ENVIRONMENT_NAME>`


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details