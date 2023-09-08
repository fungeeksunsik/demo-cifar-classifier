# demo-cifar10-classifier

Tensorflow2 offers roughly 3 code styles of model implementation: Sequential, Functional and Subclassing. This project adopts each style to implement different image classifier architecture. Specifically, three files in `models` package implements:

* feed forward architecture using Sequential API
* plain CNN architecture using functional API
* ResNet architecture using subclassing

![thumbnail](https://raw.githubusercontent.com/sunsikim/demo-cifar10-classifier/master/thumbnail.png "Main page of CIFAR10 classifier demo")

## Environment

This application is scripted and tested on M2 Apple Silicon machine.

```shell
conda env create
conda activate cifar10
```

User can try out this demo in two ways.

### 1. Download pretrained result from S3(recommended)

Each of three different pretrained models and corresponding CSV files of logs generated during training process are saved in my S3 repository. By executing `main.py` script using `remote` command, those objects can be downloaded, and you are ready to go.

```shell
python main.py remote
```

### 2. Train 3 models from scratch in local machine

Optionally, to train and save models in local machine, execute `main.py` script using `local` command. Trained models and history files will be saved in directory specified in `LOCAL_DIR` variable defined in `main.py`.

```shell
python main.py local
```

## Execute

To try out trained models, execute Streamlit demo as following.

```shell
streamlit run demo.py
```