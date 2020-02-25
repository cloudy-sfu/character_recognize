from scipy.io import loadmat
import tensorflow.keras as ks
import numpy as np


class Model:
    def __init__(self):
        _shape = (28, 28, 1)  # 28*28 像素, 1个颜色信道
        net = ks.Sequential()  # 设置模型结构
        net.add(ks.layers.Conv2D(16, kernel_size=(3, 3), activation='relu',
                                 input_shape=_shape))
        net.add(ks.layers.MaxPooling2D((2, 2)))
        net.add(ks.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
        net.add(ks.layers.MaxPooling2D((2, 2)))
        net.add(ks.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
        net.add(ks.layers.Flatten())
        net.add(ks.layers.Dense(128, activation='relu'))
        net.add(ks.layers.Dropout(0.1))
        net.add(ks.layers.Dense(47, activation="sigmoid"))  # 本层节点数一定要和字符类数相同
        net.compile(optimizer='rmsprop',
                    loss=ks.losses.sparse_categorical_crossentropy,  # loss function必须要是分类的
                    metrics=['accuracy'])
        self.net = net

    def train(self, X, Y):
        self.net.fit(X[:, :, :, np.newaxis], Y, epochs=70, batch_size=2000)  # 黑白数据X自动添加颜色信道
        self.net.save("model_1.h5")  # 保存模型

    def predict(self, X):
        Y_pred = self.net.predict(X[:, :, :, np.newaxis])  # 数据处理与train相同, 返回预测numpy.array
        Y_pred = np.argmax(Y_pred, axis=1)  # 从one-hot转换成dense形式
        return Y_pred[:, np.newaxis]  # 变成列向量, 与数据集原来的形式匹配.


if __name__ == "__main__":
    data_set = loadmat('py_emnist_balanced.mat')  # 读matlab格式数据, 拆分
    X_train, X_valid, Y_train, Y_valid = data_set['X_train'], data_set['X_test'], data_set['Y_train'], data_set['Y_test']
    X_train = X_train / 255  # 从0-255黑白空间转换到0-1黑白空间
    X_valid = X_valid / 255
    model = Model()

    model.train(X_train, Y_train)
    Y_pred = model.predict(X_valid)
    acc = np.mean(Y_pred == Y_valid)
    print("验证集准确率:", acc)
