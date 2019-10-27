import numpy as np


class NeuralNetwork:

    def __init__(self, layer_dimensions: [int], activations: list, he_initialization: bool):
        """
        Initialize the neural network model. TODO more documentation about the model

        :param layer_dimensions: List of ints, representing the number of nodes in each layer
        :param activations: List of functions, the activation function for each layer
        :param he_initialization:  Whether to use He initialization, ref
        https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e
        """
        self.layer_dimensions = layer_dimensions
        self.activations = activations
        self.he_initialization = he_initialization

        self.params = self.initialize()
        self.cost = []

    def initialize(self):
        """
        Initialize weights and biases for the neural network

        :return: A dict with W_i and b_i keys, for i in [1 until number of layers]
        """
        params = {}
        for i in range(1, len(self.layer_dimensions)):
            params['b_' + str(i)] = np.ones((self.layer_dimensions[i], 1))
            if self.he_initialization:
                params['W_' + str(i)] = np.random.randn(self.layer_dimensions[i],
                                                        self.layer_dimensions[i - 1]) * np.sqrt(
                    2 / self.layer_dimensions[i - 1])
            else:
                params['W_' + str(i)] = np.random.rand(self.layer_dimensions[i], self.layer_dimensions[i - 1]) - 0.5
        return params

    def fit(self, X, Y, learning_rate=0.1, epochs=500, verbose=1, seed: int = None):
        """

        :param X: ndarray, the design matrix
        :param Y: ndarray, response for the design matrix
        :param learning_rate: float, step size of the gradient update method
        :param epochs: number of iterations through the dataset
        :param verbose: printing level
        :param seed: seed to get consistent results
        """

        if seed is not None:
            np.random.seed(seed)

        for i in range(epochs):
            # Forward propagate
            A, temp_ = self._forward_prop(X)

            # compute cost

            # backwards propagate gradient and update weights

            # TODO print cost

    def predict(self, X):
        """
        Predict the design X and return the models probs for the result

        :param X: ndarray, design matrix to predict
        :return: model output for the provided design matrix
        """
        A, _ = self._forward_prop(X)
        return A

    def evaluate(self, X, Y):
        raise NotImplementedError

    def _forward_prop(self, X):
        temp_ = []
        A = X
        for i in range(1, len(self.layer_dimensions)):
            A_prev = A
            W = self.params['W_' + str(i)]
            b = self.params['b_' + str(i)]
            A, cache = self._dense_layer_forward(A_prev, W, b, activation=self.activations[i - 1])
            temp_.append(cache)
        return A, temp_

    @staticmethod
    def _dense_layer_forward(X, W, bias, activation):
        """
        Perform forward pass through a simple dense layer

        :param X: Inputs to the layer
        :param W: Weights for this layer
        :param bias: Bias node for this layer
        :param activation: The activation function to use
        :return: The activation vector, as well as a tuple of values
        """
        Z = np.dot(W, X) + bias
        return activation(Z)[0], (W, Z, bias)
