import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras import Sequential


class lab6:
    def __init__(self):
        self.batch_size = 500
        self.epochs = 2
        self.results = []


    def get_data(self, length = 10000):
        (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=length)
        data = np.concatenate((training_data, testing_data), axis=0)
        targets = np.concatenate((training_targets, testing_targets), axis=0)
        data = self.vectorize(data, length)
        targets = np.array(targets).astype("float32")
        test_x = data[:10000]
        test_y = targets[:10000]
        train_x = data[10000:]
        train_y = targets[10000:]
        return  train_x, train_y, test_x, test_y


    def vectorize(self, sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1
        return results


    def build_model(self):
        model = Sequential()
        model.add(layers.Dense(70, activation="relu"))
        model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
        model.add(layers.Dense(70, activation="relu"))
        model.add(layers.Dropout(0.4, noise_shape=None, seed=None))
        model.add(layers.Dense(70, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model


    def fit(self, model, length = 10000):
        model = self.build_model()
        x_train, y_train, x_test, y_test = self.get_data(length)
        history = model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                            validation_data=(x_test, y_test))
        self.results.append([length, np.mean(history.history["val_acc"])])
        return model


    def start(self):
        model = self.build_model()
        lengths = [10, 100, 500, 1000, 5000, 7000, 10000, 12000]
        for l in lengths:
            self.fit(model, l)
        self.plot_results()
        model = self.fit(model)
        self.predict_reviews(model, ["I love it, it is the best film i have ever seen", "Worst thing i have ever seen"])
        print(self.results)


    def plot_results(self):
        x = [el[0] for el in self.results]
        y = [el[1] for el in self.results]
        plt.plot(x, y)
        plt.title('Dependence of accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('length of input')
        plt.legend()
        plt.show()


    def predict_reviews(self, model, rev):
        encoded_rev = self.encode_review(rev)
        print(model.predict_classes(encoded_rev))


    def encode_review(self, rev):
        for i, el in enumerate(rev):
            el = el.split()
            rev[i] = el
            for j, word in enumerate(el):
                code = imdb.get_word_index().get(word)
                if code is None:
                    code = 0
                rev[i][j] = code
        return self.vectorize(rev)



lab = lab6()
lab.start()