# character_recognize
Classification of hand-written numbers & English characters.

`app.py` read images and predict the characters on it.

​	`test_data` a folder containing 30 images to predict.

​	`model_path` variable, the file path of neural network model, which is made via `main.py`.

​	`test_data_path` variable, the file path of images to predict.

`load_data.m` convert MATLAB dataset to Python's.

​	`emnist-balanced.mat` training data. (reference is as follow)

[Cohen, G, Afshar, S, Tapson, J, van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373]: http://arxiv.org/abs/1702.05373

`main.py` trains and valid the neural network model. 

​	`data_set` variable, the file path of MATLAB dataset, which is converted via `load_data.m`.