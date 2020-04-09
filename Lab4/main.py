import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from keras.preprocessing import image
from tensorflow.keras.optimizers import *
import pylab

class lab4:
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = mnist.load_data()
        self.train_images = self.train_images.reshape(60000, 784)
        self.test_images = self.test_images.reshape(10000, 784)
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0
        self.train_labels = to_categorical(self.train_labels)
        self.test_labels = to_categorical(self.test_labels)
        self.epochs = 7
        self.optimizers = {'adam': Adam, 'sgd': SGD, 'rmsprop': RMSprop, 'adagrad': Adagrad, 'adadelta': Adadelta}

    def read_img(self, path):
        img = image.load_img(path=path, color_mode = "grayscale", target_size=(28, 28, 1))
        img = image.img_to_array(img)
        img = 1 - img/255
        plt.imshow(img.reshape((28, 28)), cmap=plt.cm.binary)
        plt.show()
        img = img.reshape((1, 28, 28))
        return img

    def build_model(self, optimizer, opt_name, rate=0.001):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(self.train_images, self.train_labels, epochs=5, batch_size=128,
                            validation_data=(self.test_images, self.test_labels))
        x = range(1, self.epochs-1)
        plt.plot(x, history.history['loss'])
        plt.plot(x, history.history['val_loss'])
        plt.title('Model loss' + ' ({}(lrate = {}))'.format(opt_name, rate))
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        pylab.xticks(x)
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        plt.plot(x, history.history['acc'])
        plt.plot(x, history.history['val_acc'])
        plt.title('Model accuracy' + ' ({}(lrate = {}))'.format(opt_name, rate))
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        pylab.xticks(x)
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        return model

    def fimd_best_optimizer(self):
        rates = [1, 0.5, 0.1, 0.01, 0.001]
        for opt in self.optimizers.keys():
            for rate in rates:
                model = self.build_model(self.optimizers[opt](learning_rate=rate), opt, rate)


    def start(self):
        model = self.build_model(self.optimizers['adam'](), 'adam')
        #self.fimd_best_optimizer()
        image = self.read_img('0.bmp')
        res = model.predict_classes(image)
        print(res)

lab = lab4()
lab.start()