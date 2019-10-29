import matplotlib.pylab as plt
import numpy as np

from activation_functions import Sigmoid, ReLU
from nn import NeuralNetwork

network = NeuralNetwork(
    layer_dimensions=[2, 2, 1],
    activations=[ReLU, Sigmoid],
    he_initialization=True
)
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
y = np.array([[0], [1], [1], [0]]).T

network.fit(x, y, learning_rate=0.05, epochs=10000, verbose=1000)

print(network.predict(x))
plt.plot(network.cost, label='loss')
plt.plot(network.acc, label='acc')
plt.legend()
plt.show()
