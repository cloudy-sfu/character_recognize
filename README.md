# CharacterRecognition

![](https://img.shields.io/badge/build-pass-brightgreen)

This program functions as the classifier of hand-written numbers & English characters.

## Dataset

`emnist-balanced.mat`: The training data.

[Cohen, G, Afshar, S, Tapson, J, van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373]: http://arxiv.org/abs/1702.05373

## Usage

1. Run `load_data.m` to convert datasets in MATLAB format to Python format.
2. Run `main.py` to trains and valid the neural network model. 
3. Run `app.py` to read images and predict the characters on it.

|   Variable   |   Meaning   |
| ---- | ---- |
|   model_path   |   The file path of neural network model, which is made via `main.py`.   |
| test_data_path | The file path of images to predict. |
| data_set | The file path of MATLAB dataset, which is converted via `load_data.m`. |
