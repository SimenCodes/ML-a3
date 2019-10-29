import numpy as np
import matplotlib.pylab as plt

from activation_functions import ReLU, Sigmoid, Swich
from nn import NeuralNetwork

network = NeuralNetwork(
    layer_dimensions = [2, 4, 1],
    activations=[ReLU, Sigmoid],
    he_initialization=True
)
x = np.array([[0.01, 0.01], [0.01, 0.99], [0.99, 0.01], [0.99, 0.99]]).T
y = np.array([[0.01], [0.99], [0.99], [0.01]]).T

np.random.seed(0)

print(network.predict(x))

print(x.shape)
print(x)
print(y.shape)

network.fit(x, y, learning_rate=0.1, epochs=100)

print(network.predict(x))
plt.plot(network.cost)
plt.show()
