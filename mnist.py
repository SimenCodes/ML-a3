import matplotlib.pylab as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from activation_functions import Sigmoid, Swich, ReLU
from nn import NeuralNetwork

np.random.seed(0)

X, y_orig = load_digits(10, True)

y_reshaped = y_orig.reshape(y_orig.shape[0], 1)
encoder = OneHotEncoder()
# encoder.fit(np.array([range(10)]))
y = encoder.fit_transform(y_reshaped).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=24)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=24)

print(y_train)

network = NeuralNetwork(
    layer_dimensions=[X_train.shape[1], 64, 32, 16, y_train.shape[1]],
    activations=[Swich, Swich, Swich, Sigmoid],
    keep_prob=[1.0, 0.9, 0.8, 0.7, 1.0],
    he_initialization=True
)

print(X.shape, y.shape)

network.fit(X_train.T, y_train.T, X_val.T, y_val.T, learning_rate=0.005, epochs=2500, verbose=100)

# print(network.predict(X_test.T))
plt.plot(network.cost, label='loss')
plt.plot(network.val_cost, label='val_loss')
plt.plot(network.acc, label='acc')
plt.plot(network.val_acc, label='val_acc')

plt.legend()
plt.show()
