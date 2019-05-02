from ann import XOR_MLP
import numpy as np
import matplotlib.pyplot as p
from matplotlib import gridspec


def rand_input(n):
    """
    Generate randomised inputs for training
    :param n: the number of groups of 4 training values to create
    :return: the randomised inputs
    """
    x1 = np.tile([[0, 0]], (n, 1))
    x1 = x1.reshape((n, 2))
    x2 = np.tile([[0, 1]], (n, 1))
    x2 = x2.reshape((n, 2))
    x3 = np.tile([[1, 0]], (n, 1))
    x3 = x3.reshape((n, 2))
    x4 = np.tile([[1, 1]], (n, 1))
    x4 = x4.reshape((n, 2))
    x = np.concatenate((x1, x2, x3, x4))
    r = np.random.normal(0, 0.1, 8*n)
    r = r.reshape((n*4, 2))
    return x + r


if __name__ == "__main__":
    test_x = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])
    test_y = np.array([0, 1, 1, 0])
    for test_size in [4, 8, 16]:
        for hidden_nodes in [2, 4, 8]:
            x = rand_input(test_size)
            y = np.repeat([0, 1, 1, 0], test_size)
            mlp = XOR_MLP(hidden_nodes)
            mlp.train(x, y, n=500000, rate=0.1)
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
            p.title("XOR Prediction - " + str(test_size*4) + " training vectors, " + str(hidden_nodes) +
                    " hidden nodes")
            p.pcolormesh(predictions, cmap='gray', vmin=0.0, vmax=1.0)
            p.colorbar()
            p.subplot(gs[1])
            p.title("Mean squared error over epoch")
            p.ylim(0, max(mlp.error_history))
            p.plot(np.array(mlp.error_history))
            fig = p.gcf()
            p.show()
            fig.savefig(str(test_size*4) + "tv" + str(hidden_nodes) + "hn.png")
