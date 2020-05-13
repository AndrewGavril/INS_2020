import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing import sequence

class lab6:
    def __init__(self):
        self.batch_size = 70
        self.epochs = 2
        self.max_review_length = 500
        self.embedding_vecor_length = 32
        self.top_words = 5000

    def get_data(self, length=10000):
        (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=length)
        X_test = X_test[:10000]
        y_test = y_test[:10000]
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_review_length)
        X_test = sequence.pad_sequences(X_test, maxlen=self.max_review_length)
        return X_train, y_train, X_test, y_test

    def build_models(self):
        models = []
        model = Sequential()
        model.add(layers.Embedding(self.top_words, self.embedding_vecor_length, input_length=self.max_review_length))
        model.add(layers.LSTM(100))
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
        model.add(layers.Dense(70, activation="relu"))
        model.add(layers.Dropout(0.4, noise_shape=None, seed=None))
        model.add(layers.Dense(70, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        models.append(model)

        model = Sequential()
        model.add(layers.Embedding(self.top_words, self.embedding_vecor_length, input_length=self.max_review_length))
        model.add(layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
        model.add(layers.LSTM(100))
        model.add(layers.Dropout(0.25, noise_shape=None, seed=None))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        models.append(model)

        model = Sequential()
        model.add(layers.Embedding(self.top_words, self.embedding_vecor_length, input_length=self.max_review_length))
        model.add(layers.LSTM(100))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        models.append(model)

        return models

    def fit(self, models):
        x_train, y_train, x_test, y_test = self.get_data()
        for model in models:
            history = model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                                validation_data=(x_test, y_test))
            self.plot_acc(history.history['acc'], history.history['val_acc'])
        return models

    def plot_acc(self, acc, val_acc):
        plt.plot(acc, 'b', label='train')
        plt.plot(val_acc, 'r', label='validation')
        plt.title('accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epochs')
        plt.legend()
        plt.show()
        plt.clf()

    def start(self):
        models = self.build_models()
        models = self.fit(models)
        self.test_ensemble(models)
        self.predict_reviews(models[1], models[2], ["I love it, it is the best film i have ever seen",
                                                    "Worst thing i have ever seen",
                                                    "Bad producer but good actors, so the film is not so bad"])

    def test_ensemble(self, models):
        x_train, y_train, x_test, y_test = self.get_data()
        x_test = x_test[:10000]
        y_test = y_test[:10000]
        results = []
        print("Starting testing ensembles")
        for i in range(len(models)):
            for j in range(i, len(models)):
                pred = self.predict_ensemble(models[i], models[j], x_test)
                res = [1 if pred[i] == y_test[i] else 0 for i in range(len(pred))]
                res = res.count(1)/len(res)
                results.append([[i, j], res])
        print(results)

    def predict_ensemble(self, model1, model2, data):
        pred1 = model1.predict(data)
        pred2 = model2.predict(data)
        pred = [1 if (pred1[i] + pred2[i])/2 > 0.5 else 0 for i in range(len(pred1))]
        return np.array(pred)

    def predict_reviews(self, model1, model2, rev):
        encoded_rev = self.encode_review(rev)
        print(self.predict_ensemble(model1, model2, encoded_rev))

    def encode_review(self, rev):
        res = []
        for i, el in enumerate(rev):
            el = el.lower()
            delete_el = [',', '!', '.', '?']
            for d_el in delete_el:
                el = el.replace(d_el, '')
            el = el.split()
            for j, word in enumerate(el):
                code = imdb.get_word_index().get(word)
                if code is None:
                    code = 0
                el[j] = code
            res.append(el)
        for i, r in enumerate(res):
            res[i] = sequence.pad_sequences([r], maxlen=self.max_review_length)
        res = np.array(res)
        return res.reshape((res.shape[0], res.shape[2]))


lab = lab6()
lab.start()
