from os import listdir, path

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, callbacks
from keras.utils import np_utils


filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))


n_chars = len(raw_text)
n_vocab = len(chars)

print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)


class gen_text_callback(callbacks.Callback):
    def __init__(self, epochs, data, print_fun):
        super(gen_text_callback, self).__init__()
        self.epochs = epochs
        self.dataX = data
        self.print_seq = print_fun

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.epochs:
            self.print_seq(self.model, self.dataX)



class network():
    def get_data(self):
        seq_length = 100
        dataX = []
        dataY = []

        for i in range(0, n_chars - seq_length, 1):
            seq_in = raw_text[i:i + seq_length]
            seq_out = raw_text[i + seq_length]
            dataX.append([char_to_int[char] for char in seq_in])
            dataY.append(char_to_int[seq_out])

        n_patterns = len(dataX)
        print("Total Patterns: ", n_patterns)

        # reshape X to be [samples, time steps, features]
        X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
        # normalize
        X = X / float(n_vocab)
        # one hot encode the output variable
        y = np_utils.to_categorical(dataY)
        return X, y, dataX

    def build_model(self):
        X, y, dataX = self.get_data()
        model = Sequential()
        model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        epochs = [1, 10, 15]
        callbacks_list = [checkpoint, gen_text_callback(epochs, dataX, self.print_seq)]
        model.fit(X, y, epochs=35, batch_size=128, callbacks=callbacks_list)
        return model

    def print_seq(self, model, dataX):
        start = numpy.random.randint(0, len(dataX) - 1)
        pattern = dataX[start]
        print("Seed:")

        print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

        for i in range(1000):
            x = numpy.reshape(pattern, (1, len(pattern), 1))
            x = x / float(n_vocab)
            prediction = model.predict(x, verbose=0)
            index = numpy.argmax(prediction)
            result = int_to_char[index]
            print(result, end='')
            pattern.append(index)
            pattern = pattern[1:len(pattern)]

nw = network()
model =nw.build_model()

folder = '.'
filename = ''
min = 100000
for name in listdir(folder):
    full_name = path.join(folder, name)
    if path.isfile(full_name) and full_name.find('.hdf5') != -1:
        model_loss = int(full_name.split('.')[2])
        if min > model_loss:
            min = model_loss
            filename = full_name

print(filename)
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
print('FULL MODEL')
X, y, dataX = nw.get_data()
nw.print_seq(model, dataX)
print("\nDone.")