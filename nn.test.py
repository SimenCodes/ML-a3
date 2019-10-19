import unittest

from nn import *

network = initialize([2, 3])
assert network['W_1'].shape == (3, 2)
assert network['b_1'].shape == (3, 1)

A1 = dense_layer_forward(np.random.rand(2, 1), network['W_1'], network['b_1'], np.exp)
assert A1[0].shape == (3, 1)
