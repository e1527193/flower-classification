# Flower State Classification for Watering System

This project aims to develop a machine learning model to detect plants
and classify them as water-stressed or not.

## Overview of the Approach

The machine learning pipeline is divided into two stages: detection
and classification. Images are first passed to the object detection
model which draws bounding boxes where it thinks plants are. These
bounding boxes (image coordinates) allow us to get a cutout of the
image and pass said cutout to the classifier.

The classifier takes the cutout and predicts one of two classes:
*healthy* or *stressed*. The confidence (probability) of the
classifier for either class is seen as a proxy for the degree of
stress. A plant which has been classified as 70% stressed, for
example, will get a rating of 7/10 regarding its stress level, where
10 is the most severe stress. Conversely, a 60% healthy plant is put
into the 4/10 stress category.

### Plant Detection

To detect plants, we chose the popular single stage detector
YOLO. There are multiple versions available, with the most recent one
YOLOv8 available since 2023. At the start of the project YOLOv8 did
not exist yet, which is why the pipeline is built on YOLOv7.

### Classification

The classifier is a ResNet50 model which has been chosen again due to
its popularity and proven track record. It has been trained on a
dataset of only 452 healthy and 452 stressed plants, but works
surprisingly well nevertheless.

## Installing the Dependencies

In order to run the model(s), a couple of dependencies have to be
installed. These are listed in the `classification/requirements.txt`
file. The code is tested under python versions 3.6 and 3.7 and might
not work for other versions.

```bash
python3.7 -m pip install -r classification/requirements.txt
```

Afterwards, it is necessary to either add the `classification`
directory to the `PYTHON_PATH` environment variable or to install the
package locally.

```bash
# From the classification folder
python3.7 -m pip install -e .
```

## Running the Model(s)

The models can be run by calling the function `detect()` in
`evaluation/detect.py`. This will return a pandas dataframe with the
coordinates of the bounding boxes and their confidence values. This
dataframe together with the original image can be passed to the
`draw_boxes()` function in `utils/manipulations.py`. A new image with
the bounding boxes drawn on the original image can then be either
shown to the user via OpenCV or saved to disk.

## Model Evaluation

The `evaluation` directory contains multiple Jupyter notebooks which
evaluate the models separately as well as together. These notebooks
contain explanations of the code and what the results are in quasi
[literate
programming](https://en.wikipedia.org/wiki/Literate_programming)
style.

## Deploying Model to Jetson Nano

The folder `jetson-deployment` contains a script to periodically run
detections and expose the results over a simple API. The python
bindings for OpenCV are somehow very difficult to install, primarily
because NVidia does not provide regular updates to the Python and
Ubuntu distributions. Therefore, Image capture is not implemented in
Python but in C++. The binary has to be compiled with the provided
`Makefile`.
