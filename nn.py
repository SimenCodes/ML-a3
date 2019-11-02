import numpy as np
from graphviz import Digraph

import activation_functions
import cost
from layer import dense_layer_forward_with_dropout, dense_layer_backward_with_dropout, dense_layer_backward, \
    dense_layer_forward


class NeuralNetwork:

    def __init__(self, layer_dimensions: [int], activations: [activation_functions.ActivationFunction],
                 keep_prob: [float], he_initialization: bool, cost_function: cost.CostFunction = cost.CostFunction):
        """
        Initialize the neural network model. TODO more documentation about the model

        :param layer_dimensions: List of ints, representing the number of nodes in each layer
        :param activations: List of functions, the activation function for each layer
        :param he_initialization:  Whether to use He initialization, ref
        https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e
        """
        assert len(keep_prob) == len(layer_dimensions)
        assert keep_prob[0] == 1.0
        assert keep_prob[-1] == 1.0

        self.layer_dimensions = layer_dimensions
        self.activations = activations
        self.he_initialization = he_initialization
        self.cost_function = cost_function
        self.keep_prob = keep_prob

        self.params = self.initialize()
        self.cost = []
        self.acc = []
        self.val_cost = []
        self.val_acc = []

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

    def fit(self, X, Y, x_val=None, y_val=None, learning_rate=0.1, epochs=500, verbose=1, seed: int = None):
        """
        Fit the neural network to a design matrix with response

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

            _acc = self.evaluate(X, Y)
            self.acc.append(_acc)

            if x_val is not None and y_val is not None:
                A, _ = self._forward_prop(x_val)
                val_cost = self.cost_function.cost(A, y_val)
                val_acc = self.evaluate(x_val, y_val)
                self.val_cost.append(val_cost)
                self.val_acc.append(val_acc)

            if i % verbose == 0:
                if x_val is not None and y_val is not None:
                    print("Epoch: {},  Loss:{}, Acc:{}, validation_loss:{}, validation_acc:{}".format(i, _cost, _acc,
                                                                                                      val_cost,
                                                                                                      val_acc))
                else:
                    print("Epoch: {},  Loss:{}, Acc:{}".format(i, _cost, _acc))

        if x_val is not None and y_val is not None:
            print("Epoch: {},  Loss:{}, Acc:{}, validation_loss:{}, validation_acc:{}".format(epochs, _cost, _acc,
                                                                                              val_cost, val_acc))
        else:
            print("Epoch: {},  Loss:{}, Acc:{}".format(epochs, _cost, _acc))

    def predict(self, X):
        """
        Predict the design X and return the models probs for the result

        :param X: ndarray, design matrix to predict
        :return: model output for the provided design matrix
        """
        A, _ = self._forward_prop(X, training=False)
        return A

    def evaluate(self, X, Y):
        # TODO: suport arbetrary out nodes
        A = self.predict(X)

        # If pred > 0.5 Y_hat = 1 else 0
        Y_hat = np.where(A >= 0.5, 1, 0)
        acc = (1 / (Y.shape[1] * Y.shape[0])) * np.sum(np.where(Y_hat == Y, 1, 0))
        return acc

    def _forward_prop(self, X, training=True):
        cache = []
        A = X
        for i in range(1, len(self.layer_dimensions)):
            A_prev = A
            W = self.params['W_' + str(i)]
            b = self.params['b_' + str(i)]
            if training:
                A, tmp = dense_layer_forward_with_dropout(A_prev, W, b, activation=self.activations[i - 1].forward,
                                                          keep_prob=self.keep_prob[i])
            else:
                A, tmp = dense_layer_forward(A_prev, W, b, activation=self.activations[i-i].forward)
            cache.append(tmp)
        return A, cache

    def _backward_prop(self, AL, Y, cache):
        # store for future
        gradients = {}

        L = len(cache)

        # calculate grad for cost function
        dAL = self.cost_function.cost_grad(AL, Y)

        # print(self.activations[L-1])
        out = dense_layer_backward_with_dropout(dAL, cache[L - 1], self.activations[L - 1].backward,
                                                keep_prob=self.keep_prob[-1])
        gradients["dW" + str(L)], gradients["db" + str(L)], gradients["dA" + str(L)] = out

        for l in reversed(range(L - 1)):
            out = dense_layer_backward_with_dropout(gradients["dA" + str(l + 2)], cache[l],
                                                    self.activations[l].backward, keep_prob=self.keep_prob[l + 1])
            gradients["dW" + str(l + 1)], gradients["db" + str(l + 1)], gradients["dA" + str(l + 1)] = out

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

    def __repr__(self):
        return "NeuralNetwork(layer_dimensions=%s, activations=%s, he_initialization=%s, cost=%s" % (
            self.layer_dimensions, self.activations, self.he_initialization, self.cost_function
        )

    def draw(self, **kwargs):
        dot = Digraph(comment=self.__repr__())
        for layer, size in zip(range(len(self.layer_dimensions)), self.layer_dimensions):
            for node in range(size):
                id = self.node_id(layer, node)
                dot.node(id, id)
                if layer > 0:
                    prev_layer = range(self.layer_dimensions[layer - 1])
                    for prev_node in prev_layer:
                        dot.edge(self.node_id(layer - 1, prev_node), id)
        print(dot.source)
        return dot.render(**kwargs)

    def node_id(self, layer, node):
        return 'L%d-N%d' % (layer, node)
