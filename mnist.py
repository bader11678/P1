import os

import numpy as np
import gzip
import struct


def load_images(filename):
    # open and unzip the file of images:
    with gzip.open(filename, 'rb') as f:
        # read the header information into a bunch of variables:
        _ignored, n_images, columns, rows = struct.unpack('>IIII', f.read(16))
        #     read all the pixels into a numpy array of bytes:
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        #         reshape the pixels into a matrix where each line is an image
        return all_pixels.reshape(n_images, columns * rows)


def prepend_bias(X):
    return np.insert(X, 0, 1, axis=1)


mnist_dir = r"C:/Users/balsa/Documents/University/Projects/FirstProject/mnist"

X_train = prepend_bias(load_images(r"C:/Users/balsa/Documents/University/Projects/FirstProject/mnist/train-images"
                                   r"-idx3-ubyte.gz"))

X_test = prepend_bias(load_images(r"C:/Users/balsa/Documents/University/Projects/FirstProject/mnist/t10k-images-idx3"
                                  r"-ubyte.gz"))


def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(8)
        all_labels = f.read()
        return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)


def one_hot_encode(Y):
    n_labels = Y.shape[0]
    n_classes = 10
    encoded_Y = np.zeros((n_labels, n_classes))
    for i in range(n_labels):
        label = Y[i]
        encoded_Y[i][label] = 1
    return encoded_Y

Y_train_encoded = load_labels(r"C:/Users/balsa/Documents/University/Projects/FirstProject/mnist/train-labels-idx1-ubyte.gz")

Y_train = one_hot_encode(Y_train_encoded)

Y_test = one_hot_encode(load_labels(r"C:/Users/balsa/Documents/University/Projects/FirstProject/mnist/t10k-labels-idx1-ubyte.gz"))
