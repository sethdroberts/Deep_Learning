from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Sample Neural Network
network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', input_shape = (28 * 28,)))
network.add(layers.Dense(10, activation = 'softmax'))

network.compile(optimizer = 'rmsprop', 
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc: ', test_acc)

import numpy as np

#0D Tensor
x = np.array(12)
print(x)
print(x.ndim)

#1D Tensor
x = np.array([1, 2, 3, 4, 5])
print(x)
print(x.ndim)

#2D tensor
x = np.array([[1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]])
print(x)
print(x.ndim)

#3D tensor
x = np.array([[[1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]],
            [[4, 5, 6, 7, 8],
            [5, 6, 7, 8, 9],
            [6, 7, 8, 9, 10]]])

print(x)
print(x.ndim)

print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)

#Select and display a slice of a tensor
digit = train_images[4]
import matplotlib.pyplot as plt

plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

my_slice = train_images[10:100]
print(my_slice.shape)

#nth batch of a tensor
n = 0
while n < 469:
    batch = train_images[128 * n:128 * (n + 1)]
    print(batch.shape)
    n = n + 1
