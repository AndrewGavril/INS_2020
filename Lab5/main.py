import gen_data

from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt



class Net:
    def __init__(self):
        self.__set_params()
        self.x_train, self.y_train, self.num_classes,  self.input_shape, self.num_train = gen_data.get_data()


    def __set_params(self):
        self.num_epochs = 25
        self.batch_size = 100
        self.pool_size = 2
        self.conv_depth_1 = 32
        self.conv_depth_2 = 64
        self.conv_depth_3 = 128
        self.drop_prob_1 = 0.2
        self.drop_prob_2 = 0.5
        self.hidden_size = 512

    def build_model(self, kernel_size=3, hasDropout=True):
        inp = Input(shape=self.input_shape)
        conv_1 = Convolution2D(self.conv_depth_1, (kernel_size, kernel_size),
                               padding='same', strides=(1, 1), activation='relu')(inp)
        conv_2 = Convolution2D(self.conv_depth_1, (kernel_size, kernel_size),
                               padding='same', activation='relu')(conv_1)
        pool_1 = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))(conv_2)
        if hasDropout:
            drop_1 = Dropout(self.drop_prob_1)(pool_1)
        else:
            drop_1 = pool_1
        conv_3 = Convolution2D(self.conv_depth_2, (kernel_size, kernel_size),
                               padding='same', strides=(1, 1), activation='relu')(drop_1)
        conv_4 = Convolution2D(self.conv_depth_2, (kernel_size, kernel_size),
                               padding='same', strides=(1, 1), activation='relu')(conv_3)
        pool_2 = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))(conv_4)
        if hasDropout:
            drop_2 = Dropout(self.drop_prob_1)(pool_2)
        else:
            drop_2 = pool_2
        conv_5 = Convolution2D(self.conv_depth_3, (kernel_size, kernel_size),
                               padding='same', strides=(1, 1), activation='relu')(drop_2)
        conv_6 = Convolution2D(self.conv_depth_3, (kernel_size, kernel_size),
                               padding='same', strides=(1, 1), activation='relu')(conv_5)
        pool_3 = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))(conv_6)
        if hasDropout:
            drop_3 = Dropout(self.drop_prob_1)(pool_3)
        else:
            drop_3 = pool_3
        flat = Flatten()(drop_3)
        hidden = Dense(self.hidden_size, activation='relu')(flat)
        if hasDropout:
            drop_4 = Dropout(self.drop_prob_2)(hidden)
        else:
            drop_4 = hidden
        out = Dense(self.num_classes, activation='softmax')(drop_4)
        model = Model(inp, out)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def plot(self, history, hasDropout, kernel_size):
        if hasDropout:
            hasDropout = '(with dropout)'
        else:
            hasDropout = '(without dropout)'
        x = range(1, self.num_epochs + 1)
        plt.plot(x, history.history['loss'])
        plt.plot(x, history.history['val_loss'])
        plt.title('Model loss with kernel size = ' + str(kernel_size) + hasDropout)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.xlim(x[0], x[-1])
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.clf()
        plt.plot(x, history.history['accuracy'])
        plt.plot(x, history.history['val_accuracy'])
        plt.title('Model accuracy with kernel size = ' + str(kernel_size) + hasDropout)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.xlim(x[0], x[-1])
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.clf()

    def lab5(self):
        model = self.build_model()
        #history = model.fit(self.x_train, self.y_train,
        #                    batch_size=self.batch_size, epochs=self.num_epochs,
        #                    verbose=1, validation_split=0.1)
        #self.plot(history, True, 3)
        model = self.build_model(hasDropout=False)
        history = model.fit(self.x_train, self.y_train,
                            batch_size=self.batch_size, epochs=self.num_epochs,
                            verbose=1, validation_split=0.1)
        self.plot(history, False, 3)
        k_size = [2, 4, 5, 7]
        for i in k_size:
            model = self.build_model(i)
            history = model.fit(self.x_train, self.y_train,
                            batch_size=self.batch_size, epochs=self.num_epochs,
                            verbose=1, validation_split=0.1)
            self.plot(history, True, i)


lab = Net()
lab.lab5()