import numpy as np


def initialize(layers, he_init=True):
    params = {}
    for i in range(1, len(layers)):
        params['b_' + str(i)] = np.ones((layers[i], 1))
        if he_init:
            params['W_' + str(i)] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
        else:
            params['W_' + str(i)] = np.random.rand(layers[i], layers[i - 1]) - 0.5
    return params


def dense_layer_forward(X, W, bias, activation):
    Z = np.dot(W, X) + bias
    return activation(Z), (X, W, bias)

