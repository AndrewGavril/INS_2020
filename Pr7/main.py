import var2
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import matplotlib.pyplot as plt


class Net:
    def __init__(self):
        self.__set_params()
        self.train_x, self.test_x, self.val_x, self.train_y, self.test_y, self.val_y = self.get_data()


    def get_data(self):
        data, res = var2.gen_data_from_sequence()
        dataset_size = len(data)
        train_size = (dataset_size // 10) * 7
        val_size = (dataset_size - train_size) // 2
        train_data, train_res = data[:train_size], res[:train_size]
        val_data, val_res = data[train_size:train_size + val_size], res[train_size:train_size + val_size]
        test_data, test_res = data[train_size + val_size:], res[train_size + val_size:]
        return train_data, test_data, val_data, train_res, test_res, val_res

    def __set_params(self):
        self.num_epochs = 15
        self.batch_size = 5

    def build_model(self):
        model = Sequential()
        model.add(layers.GRU(32, recurrent_activation='sigmoid', input_shape=(None, 1), return_sequences=True))
        model.add(layers.LSTM(16, activation='relu', input_shape=(None, 1), return_sequences=True, dropout=0.2))
        model.add(layers.GRU(32, input_shape=(None, 1), recurrent_dropout=0.2))
        model.add(layers.Dense(1))
        model.compile(optimizer='nadam', loss='mse')
        return model

    def plot(self, history):
        x = range(1, self.num_epochs + 1)
        plt.plot(x, history.history['loss'])
        plt.plot(x, history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


    def pr7(self):
        model = self.build_model()
        history = model.fit(self.train_x, self.train_y, epochs=self.num_epochs, validation_data=(self.val_x, self.val_y))
        self.plot(history)
        res = model.predict(self.test_x)
        self.draw_seq(res, self.test_y)

    def draw_seq(self, seq1, seq2):
        plt.plot(range(len(seq1)), seq1)
        plt.plot(range(len(seq2)), seq2)
        plt.show()

pr = Net()
pr.pr7()