# Facial Expression Analyser
  
## Introduction
* Function: provides real-time analysis on facial expression upon each successful face detection through video cameras.
* Implementation: the detector was trained and fine-tuned by applying supervised learning on fer2013 datasets via convolutional neural networks backended in TensorFlow.
* Status: work in progress

![alt text](https://camo.githubusercontent.com/1447b41176420ee6ca0232ea41891cf3af907be2/68747470733a2f2f692e696d6775722e636f6d2f434c37344a36352e706e67)

## Scripts, Models & Datasets
Facial Expression Analyser contains several training/testing scripts and pre-trained models for you to play with.

**Scripts**

Name | Usage
------------ | -------------
main.py | - This is the original script, this script serves as the starting point of the program <br> - You can set different arguments within the main function for different purposes (See Instructions)
faceDetectionFunc.py | This script contains algorithms for face detection
facialExpressionFunc.py | This script is the brain to facial expression detection

**Datasets**

Name | Usage
------------ | -------------
fer2013 | - Kaggle dataset that contains 7 different facial expressions

**Pre-trained Model Weights**

Name | Usage
------------ | -------------
model_0/facial_expression_model_weights.h5 | The DEMO weight
model_(1,2,3)/facial_expression_model_weights.h5 | The weight used during testing

## To run it:
* Get Python3, Dilb, OpenCV, numpy, Keras & Tensorflow, and ur ready to rock'n roll :) (I guess you need a camera to say cheese)





