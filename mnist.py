import matplotlib.pylab as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from activation_functions import ReLU, Sigmoid
from nn import NeuralNetwork

X, y_orig = load_digits(10, True)

y_reshaped = y_orig.reshape(y_orig.shape[0], 1)
encoder = OneHotEncoder()
# encoder.fit(np.array([range(10)]))
y = encoder.fit_transform(y_reshaped).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=24)

print(y_train)

network = NeuralNetwork(
    layer_dimensions=[X_train.shape[1], 4, y_train.shape[1]],
    activations=[ReLU, Sigmoid],
    he_initialization=True
)

print(X.shape, y.shape)

network.fit(X_train.T, y_train.T, learning_rate=0.2, epochs=2500, verbose=100)

print(network.predict(X_test))
plt.plot(network.cost, label='loss')
plt.plot(network.acc, label='acc')
plt.legend()
