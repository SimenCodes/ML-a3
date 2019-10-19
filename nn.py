import numpy as np


def initialize(layers, he_init=True):
    """
    Initialize weights and biases for the neural network
    :param layers: List of ints, representing the number of nodes in each layer
    :param he_init: Whether to use He initialization, ref https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e
    :return: A dict with W_i and b_i keys, for i in [1..number of layers]
    """
    params = {}
    for i in range(1, len(layers)):
        params['b_' + str(i)] = np.ones((layers[i], 1))
        if he_init:
            params['W_' + str(i)] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
        else:
            params['W_' + str(i)] = np.random.rand(layers[i], layers[i - 1]) - 0.5
    return params


def dense_layer_forward(X, W, bias, activation):
    """
    Perform forward pass through a simple dense layer
    :param X: Inputs to the layer
    :param W: Weights for this layer
    :param bias: Bias node for this layer
    :param activation: The activation function to use
    :return: The activation vector, as well as a copy of the input values
    """
    z = np.dot(W, X) + bias
    return activation(z), (X, W, bias)


def cost(Y_pred, Y_expected):
    """
    Find out how wrong we were
    :param Y_pred:
    :param Y_expected:
    :return: The Mean Squared Error
    """
    return 0.5 * ((Y_pred - Y_expected) ** 2)
