import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

class pr4:
    def __init__(self):
        self.data = np.array([[1, 2, 1],
                    [2, 2, 2]])
        self.model = self.get_model()


    def operation(self, x):
        return (x[0] or x[1]) ^ (not(x[1] and x[2]))

    def get_model(self):
        model = Sequential()
        model.add(Dense(4, activation='relu', input_shape=(3,)))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def relu(self, x):
        return np.maximum(x, 0)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sim_by_el(self):
        weights = [layer.get_weights() for layer in self.model.layers]
        data = self.data.copy()
        for l in range(0, len(weights)):
            res = np.zeros((data.shape[0], weights[l][0].shape[1]))
            for i in range(0, data.shape[0]):
                for j in range(0, weights[l][0].shape[1]):
                    s = 0
                    for k in range(0, data.shape[1]):
                        s += data[i][k] * weights[l][0][k][j]
                    if l < len(weights)-1:
                        res[i][j] = self.relu(s + weights[l][1][j])
                    else:
                        res[i][j] = self.sigmoid(s + weights[l][1][j])
            data = res
        return res

    def sim_by_numpy(self):
        res = self.data.copy()
        weights = [layer.get_weights() for layer in self.model.layers]
        for i in range(len(weights)-1):
            res = self.relu(np.dot(res, weights[i][0]) + weights[i][1])
        res = self.sigmoid(np.dot(res, weights[-1][0]) + weights[-1][1])
        return res

    def get_res(self):
        return [self.model.predict(self.data), self.sim_by_el(), self.sim_by_numpy()]

    def print_res(self, model, el, numpy):
        print("Model res:\n{}".format(model))
        print("Operations with elements res:\n{}".format(el))
        print("Using numpy res:\n{}".format(numpy))

    def start(self):
        [a, b, c] = self.get_res()
        assert np.isclose(a, b).all()
        assert np.isclose(a, c).all()
        self.print_res(a, b, c)
        train_res = np.array([int(self.operation(x)) for x in self.data])
        self.model.fit(self.data, train_res, epochs=150, batch_size=1)

        [a, b, c] = self.get_res()
        assert np.isclose(a, b).all()
        assert np.isclose(a, c).all()
        self.print_res(a, b, c)

practice = pr4()
practice.start()

