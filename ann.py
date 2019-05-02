import numpy as np
import matplotlib.pyplot as p
from matplotlib import gridspec


def sigmoid(x):
    """
    The sigmoid activation function
    :param x: the input value
    :return: the sigmoid function of the input
    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    """
    Derivative of the sigmoid activation function
    :param x: the input value
    :return: the derivative of the sigmoid function of the input
    """
    return sigmoid(x) * (1.0 - sigmoid(x))


def rand_m(s):
    """
    Generate a random numpy matrix of values between -1 and 1
    :param s: the shape of the matrix
    :return: the random vector
    """
    return 2*np.random.random(s) - 1


class XOR_MLP:
    """
    MLP class specifically for the XOR problem.
    The following website was used as a reference for implementation:
    https://www.bogotobogo.com/python/python_Neural_Networks_Backpropagation_for_XOR_using_one_hidden_layer.php
    """

    def __init__(self, n):
        """
        Instantiate the XOR MLP
        :param n: the number of nodes in the hidden layer
        """
        self.error_history = []
        self.w = []  # the weights
        self.w.append(rand_m((3, n+1)))  # 2 + 1 -> n + 1
        self.w.append(rand_m((n+1, 1)))  # n + 1 -> 1

    def train(self, x, y, n=10000, rate=0.1, threshold=0.0):
        """
        Train the XOR MLP
        :param x: the input training data
        :param y: the expected outputs for the training data
        :param n: the number of iterations of training
        :param rate: the rate at which deltas are updated
        :param threshold: the squared error upon which to stop training
        """
        ones = np.atleast_2d(-np.ones(x.shape[0]))
        x = np.concatenate((ones.T, x), axis=1)
        self.error_history = []

        for i in range(n):
            # select a random example from the training data
            r = np.random.randint(x.shape[0])
            a = [x[r]]

            for layer in range(len(self.w)):
                product = np.dot(a[layer], self.w[layer])
                a.append(sigmoid(product))

            error = y[r] - a[-1]  # difference between output and prediction
            if len(self.error_history) > 0:
                prev_err = self.error_history[-1]
            else:
                prev_err = 0
            err_sq = error ** 2
            mse = ((prev_err * i) + err_sq) / (i+1)
            self.error_history.append(mse)
            deltas = [error * sigmoid_prime(a[-1])]

            # calculate forward-pass deltas
            for layer in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.w[layer].T) * sigmoid_prime(a[layer]))

            deltas.reverse()

            # perform backpropagation
            for j in range(len(self.w)):
                layer = np.atleast_2d(a[j])
                delta = np.atleast_2d(deltas[j])
                self.w[j] += rate * layer.T.dot(delta)

            # halt training if difference in squared error is less than threshold
            if abs(err_sq - prev_err) < threshold:
                break

    def predict(self, x):
        """
        Predict using the MLP
        :param x: input data
        :return: prediction
        """
        x = np.concatenate(np.atleast_2d((-np.ones(1)).T, np.array(x)), axis=1)
        for i in range(len(self.w)):
            x = sigmoid(np.dot(x, self.w[i]))
        return x


if __name__ == "__main__":
    mlp = XOR_MLP(8)
    x = np.array([[-0.310691, 0.0164278],
                  [-0.309003, 0.898471],
                  [1.25774, -0.231735],
                  [1.31959, 0.82952],
                  [-0.0897083, -1.02045],
                  [-0.457115, 1.84369],
                  [1.42524, 0.111823],
                  [1.43962, 0.28365],
                  [-0.21377, 0.0759174],
                  [-0.16744, 0.985518],
                  [0.579612, 0.584378],
                  [1.90558, 0.434351],
                  [0.442017, 0.35245],
                  [0.204012, -0.0194183],
                  [1.75664, -0.336488],
                  [0.584128, 1.45608]])
    y = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    test_x = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])
    test_y = np.array([0, 1, 1, 0])
    mlp.train(x, y, n=500000)
    for e in test_x:
        print(e, mlp.predict(e))
    z = []
    for i in np.arange(0, 1.01, 0.05):
        t = []
        for j in np.arange(0, 1.01, 0.05):
            t.append(mlp.predict([i, j])[0][0])
        z.append(t)
    predictions = np.array(z)
    fig = p.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    p.subplot(gs[0])
    p.title("XOR Prediction - example training data set")
    p.pcolormesh(predictions, cmap='gray', vmin=0.0, vmax=1.0)
    p.colorbar()
    p.subplot(gs[1])
    p.title("Mean squared error over epoch")
    p.ylim(0, max(mlp.error_history))
    p.plot(np.array(mlp.error_history))
    fig = p.gcf()
    p.show()
