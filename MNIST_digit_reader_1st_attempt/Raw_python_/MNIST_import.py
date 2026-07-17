import kagglehub
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import struct
from array import array
from os.path  import join

# Path to dataset files: C:\Users\User\.cache\kagglehub\datasets\hojjatk\mnist-dataset\versions\1

training_images_filepath = r"C:\Users\User\.cache\kagglehub\datasets\hojjatk\mnist-dataset\versions\1\train-images-idx3-ubyte\train-images-idx3-ubyte"
training_labels_filepath = r"C:\Users\User\.cache\kagglehub\datasets\hojjatk\mnist-dataset\versions\1\train-labels-idx1-ubyte\train-labels-idx1-ubyte"
test_labels_filepath     = r"C:\Users\User\.cache\kagglehub\datasets\hojjatk\mnist-dataset\versions\1\t10k-labels-idx1-ubyte\t10k-labels-idx1-ubyte" 
test_images_filepath     = r"C:\Users\User\.cache\kagglehub\datasets\hojjatk\mnist-dataset\versions\1\t10k-images-idx3-ubyte\t10k-images-idx3-ubyte"
import kagglehub

# Download latest version
path = kagglehub.dataset_download("hojjatk/mnist-dataset")

print("Path to dataset files:", path)

# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = np.frombuffer(file.read(), dtype=np.uint8)      
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            images = np.frombuffer(file.read(), dtype=np.uint8).reshape(size, rows, cols)
    
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)  
    
# 1. Initialize and load
dataloader = MnistDataloader(
    training_images_filepath,
    training_labels_filepath,
    test_images_filepath,
    test_labels_filepath,
)

# Display matrix data with a gray or customized colormap
#print(y_train[2])

