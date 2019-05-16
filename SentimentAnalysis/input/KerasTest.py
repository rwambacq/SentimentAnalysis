import keras
import sklearn
import numpy as np
from sklearn.datasets import load_breast_cancer

from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(units=64, activation='relu', input_dim=30),
    Dense(units=10, activation='relu'),
    Dense(units=2, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


wisconsin = load_breast_cancer()
x_data = wisconsin['data']
y_data = wisconsin['target']

print('Aantal samples: {}'.format(x_data.shape[0]))
print('Aantal features: {}'.format(x_data.shape[1]))

class0 = x_data[y_data == 0].shape[0]
class1 = x_data[y_data == 1].shape[0]
ratio = max([class0, class1]) / min([class0, class1])
print('Klasse 0: {}'.format(class0))
print('Klasse 1: {}'.format(class1))
print('Imbalance ratio: {}'.format(ratio))

y_data = np.array([[1., 0.] if y == 0 else [0., 1.] for y in y_data])

from sklearn.utils import shuffle

x_data, y_data = shuffle(x_data, y_data)

p = .8
idx = int(x_data.shape[0] * p)
x_train, y_train = x_data[:idx], y_data[:idx]
x_test, y_test = x_data[idx:], y_data[idx:]

x_mean, x_std = x_train.mean(), x_train.std()
x_train -= x_mean
x_train /= x_std

x_test -= x_mean
x_test /= x_std

model.fit(x_train, y_train, epochs=100, batch_size=65)

classes = np.argmax(model.predict(x_test, batch_size=65), axis=1)

accuracy = np.mean(np.equal(classes, np.argmax(y_test, axis=1)))
print('Accuracy: {}'.format(accuracy))

evals = model.evaluate(x_test, y_test, batch_size=65)
print('Loss: {}'.format(evals[0]))
print('Accuracy: {}'.format(evals[1]))

labels = np.argmax(y_test, axis=1)
idx0 = (labels == 0)
idx1 = (labels == 1)

acc0 = np.mean(np.equal(classes[idx0], labels[idx0]))
acc1 = np.mean(np.equal(classes[idx1], labels[idx1]))
bal_acc = (acc0 + acc1) / 2
print('Balanced accuracy: {}'.format(bal_acc))
print('\tKlasse 0: {}'.format(acc0))
print('\tKlasse 1: {}'.format(acc1))