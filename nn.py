import numpy as np
import cost
import activation_functions
import matplotlib.pylab as plt

class NeuralNetwork:

    def __init__(self, layer_dimensions: [int], activations: [activation_functions.ActivationFunction],
                 he_initialization: bool, cost_function: cost.CostFunction = cost.CostFunction):
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
        self.cost_function = cost_function

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
            A, cache = self._forward_prop(X)

            # compute cost
            _cost = self.cost_function.cost(A, Y)

            # backwards propagate gradient and update weights
            gradients = self._backward_prop(A, Y, cache)

            # Update parameters
            self._update_params(gradients, learning_rate)

            # save cost for history
            self.cost.append(_cost)
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
        cache = []
        A = X
        for i in range(1, len(self.layer_dimensions)):
            A_prev = A
            W = self.params['W_' + str(i)]
            b = self.params['b_' + str(i)]
            A, tmp = self._dense_layer_forward(A_prev, W, b, activation=self.activations[i - 1].forward)
            # print("z_"+str(i) + " " + str(A))
            cache.append(tmp)
        return A, cache

    def _backward_prop(self, AL, Y, cache):
        # store for future
        gradients = {}

        L = len(cache)

        # calculate grad for cost function
        dAL = self.cost_function.cost_grad(AL, Y)

        # print(self.activations[L-1])
        out = self._dense_layer_backward(dAL, cache[L - 1], self.activations[L - 1].backward)
        gradients["dW" + str(L)], gradients["db" + str(L)], gradients["dA" + str(L)] = out

        for l in reversed(range(L - 1)):
            out = self._dense_layer_backward(gradients["dA" + str(l + 2)], cache[l], self.activations[l].backward)
            gradients["dW" + str(l+1)], gradients["db" + str(l+1)], gradients["dA" + str(l+1)] = out

        return gradients

    def _update_params(self, gradients: dict, learning_rate: float):
        """
        Update the parameters of the model

        :param gradients: gradients from back-propagation
        :param learning_rate: step size of the gradient update
        """
        L = len(self.activations)

        for l in range(L):
            self.params["W_" + str(l + 1)] = self.params["W_" + str(l + 1)] - learning_rate * gradients[
                "dW" + str(l + 1)]
            self.params["b_" + str(l + 1)] = self.params["b_" + str(l + 1)] - learning_rate * gradients[
                "db" + str(l + 1)]

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
        return activation(Z)[0], (X, W, Z, bias)

    @staticmethod
    def _dense_layer_backward(g, old_values, activation_backwards):
        """
        Compute gradients for this layer
        :param g: Gradients propagated from the next layer
        :param old_values: the X, W, Z, and bias nodes
        :param activation_backwards: the backwards implementation of the activation function
        :return: weight and bias changes, and the gradients to propagate to the previous layer
        """
        X, W, Z, bias = old_values
        g = activation_backwards(g, Z)
        dW = np.dot(g, X.T)  # / m
        dB = np.mean(g)
        g = np.dot(W.T, g)
        return dW, dB, g


if __name__ == '__main__':
    np.random.seed(0)
    network = NeuralNetwork([2,4, 1], [activation_functions.ReLU,activation_functions.Sigmoid], True)
    x = np.array([[0.01, 0.01], [0.01, 0.99], [0.99, 0.01], [0.99, 0.99]]).T
    y = np.array([[0.01], [0.99], [0.99], [0.01]]).T

    print(network.predict(x))

    print(x.shape)
    print(x)
    print(y.shape)

    network.fit(x, y, learning_rate=0.1, epochs=100)

    print(network.predict(x))
    plt.plot(network.cost)
    plt.show()