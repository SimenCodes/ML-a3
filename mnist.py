import matplotlib.pylab as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from activation_functions import Sigmoid
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
    layer_dimensions=[X_train.shape[1], 32, 16, y_train.shape[1]],
    activations=[Sigmoid, Sigmoid, Sigmoid],
    keep_prob=[1.0, 0.8, 0.9, 1.0],
    # keep_prob=[1.0, 0.8, 0.7, 1.0],
    he_initialization=True
)

print(X.shape, y.shape)

network.draw(filename='mnist', format='png')

network.fit(X_train.T, y_train.T, X_val.T, y_val.T, learning_rate=0.1, epochs=25000, verbose=100)

y_val_pred = np.argmax(network.predict(X_val.T), axis=0)
cm = confusion_matrix(np.argmax(y_val.T, axis=0), y_val_pred, labels=list(range(10)))

print(cm)

plt.imshow(cm)
plt.xlabel("predicted")
plt.ylabel("true")
plt.xticks(list(range(10)))
plt.yticks(list(range(10)))
plt.ylim([-0.5, 9.5])
plt.show()

# print(network.predict(X_test.T))
plt.plot(network.cost, label='loss')
plt.plot(network.val_cost, label='val_loss')
plt.plot(network.acc, label='acc')
plt.plot(network.val_acc, label='val_acc')
plt.legend()
plt.show()

pred = np.argmax(network.predict(X_test.T), axis=0)
cm = confusion_matrix(np.argmax(y_test.T, axis=0), pred, labels=list(range(10)))

print(cm)

plt.imshow(cm)
plt.xlabel("predicted")
plt.ylabel("true")
plt.xticks(list(range(10)))
plt.yticks(list(range(10)))
plt.ylim([-0.5, 9.5])
plt.show()

print(network.evaluate(X_test.T, y_test.T))
