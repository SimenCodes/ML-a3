import numpy as np


def sigmoid(Z):
    """
    Sigmoid activation function

    :param Z: numpy array of any shape
    :return: output from sigmoid, same shape as Z, Z usefull for backprop
    """
    A = 1 / (1 + np.exp(-Z))
    return A, Z


def sigmoid_backward(dA, old_value):
    """
    Sigmoid activation function backwards

    :param dA: post-activation gradient, of any shape
    :param old_value: old z to efficiently compute the backward function
    :return: gradient of the cost with respect to Z
    """
    Z = old_value

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert (dZ.shape == Z.shape)
    return dZ


def relu(Z):
    """
    Relu activation function

    :param Z: numpy array of any shape
    :return: output from sigmoid, same shape as Z, Z usefull for backprop
    """
    A = np.maximum(0, Z)
    return A, Z


def relu_backward(dA, old_value):
    """
    Relu activation function backwards

    :param dA: post-activation gradient, of any shape
    :param old_value: old z to efficiently compute the backward function
    :return: gradient of the cost with respect to Z
    """
    Z = old_value
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0
    return dZ


def swich(Z):
    """
    Swich activation function, ref https://arxiv.org/pdf/1710.05941v1.pdf

    :param Z: numpy array of any shape
    :return: output from sigmoid, same shape as Z, Z usefull for backprop
    """
    A = Z / (1 + np.exp(-Z))
    return A, Z


def swich_backward(dA, old_value):
    """
    Swich activation function backwards

    :param dA: post-activation gradient, of any shape
    :param old_value: old z to efficiently compute the backward function
    :return: gradient of the cost with respect to Z
    """
    Z = old_value
    s = 1 / (1 + np.exp(-Z))
    f = s * Z
    dZ = (f + s * (s - f)) * dA
    return dZ
