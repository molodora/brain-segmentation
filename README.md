# Brain Segmentation App

## Overview

A backend web server built with FastAPI, leveraging a [U-Net](https://arxiv.org/abs/1505.04597) model for segmenting brain MR images. The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).

## Project Structure
```
├── app                       <- Fast API app
│   ├── app.py                <- App script
│   └── unet-scripted.pt      <- Model
│
├── data                      <- Project data
│
├── models                    <- Models and checkpoints
│
├── src                       <- Source code
│   ├── data                  <- Data scripts
│   ├── models                <- Model scripts
│   │
│   ├── train.py              <- Training script
│   └── utils.py              <- Utilities script
│
├── tests                     <- Pyteset tests
│
├── .dockerignore             <- List of files ignored by docker
├── .gitignore                <- List of files ignored by git
├── Dockerfile                <- File for building a docker image
├── requirements.txt          <- Python dependencies for app
└── README.md
```

It is assumed that the trained model in TorchScript format will be located in `app/`. To train the model, configure the training parameters in the `train.py` file and execute it. Move the trained model from the `models/` folder to `app/`.

## Run the App in Docker Container

1. Build docker image
```
docker build -t brain_app .
```
2. Run docker container
```
docker run --rm -d --name brain_app -p 8000:8000 brain_app
```
3. Now you can send a POST request with an image, for example, using the python requests library:
```python
import requests
from pathlib import Path

response = requests.post(
    "http://localhost:8000/predict", files = {"file": open(Path(__file__).parent / "img.tif", "rb")}
)
with open(Path(__file__).parent /'test.png', 'wb') as f:
    f.write(response.content)
```
