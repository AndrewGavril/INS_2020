from tensorflow.keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np

def get_data(size = None):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    num_train, depth, height, width = x_train.shape
    num_test = x_test.shape[0]
    num_classes = np.unique(y_train).shape[0]
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    input_shape = (depth, height, width)
    return x_train, y_train, num_classes, input_shape, num_train