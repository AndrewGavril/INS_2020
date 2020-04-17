from var2 import gen_data
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.losses import BinaryCrossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class pr6:
    def __init__(self):
        self.batch_size = 100
        self.num_classes = 2
        self.epochs = 75
        self.img_size = 28
        self.x_train, self.y_train, self.x_test, self.y_test = self.get_data()

    def get_data(self):
        x, y = gen_data(1000, 28)
        x, y = np.asarray(x), np.asarray(y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train = x_train.reshape(x_train.shape[0], self.img_size, self.img_size, 1)
        x_test = x_test.reshape(x_test.shape[0], self.img_size, self.img_size, 1)
        x_train /= 255
        x_test /= 255

        encoder = LabelEncoder()
        encoder.fit(y_train)
        y_train = encoder.transform(y_train)
        y_train = to_categorical(y_train, self.num_classes)

        encoder.fit(y_test)
        y_test = encoder.transform(y_test)
        y_test = to_categorical(y_test, self.num_classes)

        return x_train, y_train, x_test, y_test

    def build_model(self):
        input_shape = (self.img_size, self.img_size, 1)
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape, name='first'))
        model.add(Conv2D(64, (3, 3), activation='relu', name='conv2_2'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='mp2d_3'))
        model.add(Dropout(0.25, name='first_dropout'))
        model.add(Flatten(name='flatten'))
        model.add(Dense(128, activation='relu', name='simple_dense'))
        model.add(Dropout(0.5, name='second_dropout'))
        model.add(Dense(self.num_classes, activation='softmax', name='last_layer'))

        model.compile(Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])
        H = model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True,
            validation_data=(self.x_test, self.y_test),
        )
        self.plot(H, self.epochs)
        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return model

    def plot(self, history, epochs):
        loss = history.history['loss']
        v_loss = history.history['val_loss']
        x = range(1, epochs + 1)
        plt.plot(x, loss, 'b', label='train')
        plt.plot(x, v_loss, 'r', label='validation')
        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend()
        plt.show()
        plt.clf()

    def start(self):
        model = self.build_model()
        test, label = gen_data(2, 28)
        plt.imshow(test[0].reshape((28, 28)), cmap=plt.cm.binary)
        plt.show()
        test = np.asarray(test)
        test = test.astype('float32')
        test /= 255
        print(model.predict_classes(test[0].reshape((1, 28, 28, 1))))
        print(label)

pract = pr6()
pract.start()