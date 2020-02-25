import tensorflow.keras as ks
import os
import cv2
import numpy as np


def cv_process(fn):
    img = cv2.imread(fn, 0)
    img = 1 - img / 255
    img = cv2.dilate(img, kernel=np.ones((7, 7)))
    img = cv2.resize(img, dsize=(28, 28))
    return img[:, :, np.newaxis]


model_path = 'model_3.h5'
test_data_path = 'test_data'
predict_dict = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b',
                'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
net = ks.models.load_model(model_path)
test_data_name = os.listdir(test_data_path)
test_data = [
    cv_process(test_img) for test_img in map(lambda x: test_data_path + '/' + x, test_data_name)
    # map 变绝对路径
    # for 每一个文件名, 输出预处理结果, 放在List里.
]
test_data = np.array(test_data)
test_label = net.predict(test_data)
test_label = np.argmax(test_label, axis=1)
test_label = [
    predict_dict[i] for i in test_label
]
for x, y in zip(test_data_name, test_label):
    print("图片地址:", x, "预测结果:", y)
