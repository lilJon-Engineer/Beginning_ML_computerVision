import numpy as np
import math
import time
from pathlib import Path
from MNIST_import import MnistDataloader
import seaborn as sns
import matplotlib.pyplot as plt
import torch

training_images_filepath = r"C:\Users\User\.cache\kagglehub\datasets\hojjatk\mnist-dataset\versions\1\train-images-idx3-ubyte\train-images-idx3-ubyte"
training_labels_filepath = r"C:\Users\User\.cache\kagglehub\datasets\hojjatk\mnist-dataset\versions\1\train-labels-idx1-ubyte\train-labels-idx1-ubyte"
test_labels_filepath     = r"C:\Users\User\.cache\kagglehub\datasets\hojjatk\mnist-dataset\versions\1\t10k-labels-idx1-ubyte\t10k-labels-idx1-ubyte" 
test_images_filepath     = r"C:\Users\User\.cache\kagglehub\datasets\hojjatk\mnist-dataset\versions\1\t10k-images-idx3-ubyte\t10k-images-idx3-ubyte"
dataloader = MnistDataloader(
    training_images_filepath,
    training_labels_filepath,
    test_images_filepath,
    test_labels_filepath,
)
(x_train, y_train), (x_test, y_test) = dataloader.load_data()

start = time.time()
d = []
for i in range(len(y_train)):
    c = ([0]*10)
    c[y_train[i]] = 1
    d.append(c)
#print("Length of d (number of solutions):", len(d), "Ex: Solution:" + str(d[3]) + ". Compare solution to y_train[3]: " + str(y_train[3]))
end = time.time()
print("Time taken to create d:", end - start)

# Initialize size of first hidden layer of the neural network
alpha = 0.005
layer_1_size = len(x_train[0])*len(x_train[0][0]) + 1  # Size of the first hidden layer is equal to the number of pixels in the input image (28x28 = 784)
output_layer_size = 10
numLayers = 1 
# Find a method for initializing the weights of the neural network. Bounds, based on Le Cun (1986) [-2.4/F, 2.4/F] or [-2.4/sqrt(F), 2.4/sqrt(F)]
layer_1_weight_range = 2.4 / math.sqrt(layer_1_size) # Le Cun (1986) weight initialization bounds
weights = np.random.uniform(-layer_1_weight_range, layer_1_weight_range, size=(numLayers, output_layer_size, layer_1_size))  # Initialize weights for the first hidden layer

# Variables for the scaled hyperbolic tangent function
A = 1.7159
S = 2/3

print("First weights index size:" , len(weights))
print("Second weights index size:" , len(weights[0]))
print("Last weights index size:", len(weights[0][0][:]))

def dot(a, b): # Returns the dot product of two vectors a and b. a and b must be the same length.
    if len(a) != len(b):
        raise ValueError("Vectors must be the same length")
    return sum(x * y for x, y in zip(a, b))

def image2vector(image): # Converts a 2D image into a 1D column vector
    c = []
    for row in range(len(image)):
        for col in range(len(image[row])):
            c.append(image[row][col] / 255.0)   # normalize
    return c

def vector_difference_RMS(a, b): # Returns the sum of the differences between two vectors a and b. a and b must be the same length.
    if len(a) != len(b):
        raise ValueError("Vectors must be the same length")
    return sum(abs(x - y)**2 for x, y in zip(a, b))

def vector_difference(a, b): # Returns the difference vector between vectors a and b. a and b must be the same length.
    if len(a) != len(b):
        raise ValueError("Vectors must be the same length")
    return [x - y for x, y in zip(a, b)]

def sech(x):
    try:
        return 1 / math.cosh(x)
    except OverflowError:
        # If cosh(x) overflows, it means cosh(x) is effectively infinity.
        # 1 / infinity is 0.
        return 0.0
    
def weights_difference(a, b): # Returns the difference between two matrices a and b. a and b must be the same size.
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("Matrices must be the same size")
    return [[a[i][j] - alpha * b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

# Compute the number of dot product between the input and the weights of the first hidden layer for each neuron. Time this process to understand how scalable the neural network is.

# Pass each of the neuron dot products through the scaled hyperbolic tangent function to get the output of the first hidden layer.
# Time this process to understand how scalable the neural network is.

print("Layer 1 Size:", layer_1_size)

data_folder = Path(r"C:\Users\User\OneDrive\Desktop\ML_number_reader_1st_paper_tests\Beginning_ML_computerVision\MNIST_digit_reader_1st_attempt\Raw_python_\Data")
error_file_path = data_folder / "error_file.txt"
alpha = 0.05 # Learning rate
grad_C = np.zeros((output_layer_size, layer_1_size))  # Initialize the gradient of the cost function with respect to the weights of the first layer
epochs = 4 # Number of iterations of SGD

# Create the file if it does not exist
error_file_path.touch(exist_ok=True)
error_file_path.write_text("")   # reset before starting a new full training run
print("This will take around 30 minutes to 3 hours to complete if you're training on the full 60k images")
with open(error_file_path, "a") as file:
    for z in range(epochs):
        start = time.time()
        for c in range(len(x_train)):
            input_vector = image2vector(x_train[c])  # Convert the input image to a vector
            input_vector.append(1)
            # For each datapoint in the training set, compute the dot prdouct between the input image and the weights of the first hidden layer, 
            # and then pass the result through the scaled hyperbolic tangent function to get the output of the first hidden layer.
            net_input = [dot(input_vector, weights[0][j]) for j in range(output_layer_size)]  # Compute the dot product between the input image and the weights of the first hidden layer for each neuron
            output_layer = [A * math.tanh(S * net_input[j]) for j in range(output_layer_size)]  # Compute the output of the first hidden layer for the first training image
            file.write(str(vector_difference_RMS(output_layer, d[c]) / (2)) + "\n")   # Compute the error between the output of the first hidden layer and the expected output for the first training image
            # Compute the gradient of the cost function with respect to the weights of the first hidden layer ( This is a matrix )
            grad_C = [[x * input_vector[i] for i in range(layer_1_size)] for x in [(output_layer[j] - d[c][j]) * S * A * sech(S * net_input[j]) ** 2 for j in range(output_layer_size)]]
            weights[0] = weights_difference(weights[0], grad_C)  # Update the weights of the input to output layer using the negative gradient of the cost function, with a prop. const. alpha (learning rate) to control the step size of the update.
        file.write("New Epoch " + str(z) + "\n\n")  # Add a newline after each epoch
        file.flush() # Save the file after each epoch to ensure that the data is not lost in case of a crash or interruption.
        print("Time for Epoch", z, ":", time.time() - start)
# checkLater: need to change c to the range of the length of x_train to compute the output for all training images, but this will take a long time to compute. For now, we will just compute the output for the first 10 training images.
# about 9.5 ms per dot product, so about 9.5 min for 10,000 images.
print("Time taken to compute output_layer:", time.time() - start)

confusion_matrix = [[0 for _ in range(len(output_layer))] for _ in range(len(output_layer))]
correct = 0
for c in range(len(x_test)):
    input_vector = image2vector(x_test[c])  # Convert the input image to a vector
    input_vector.append(1)
    net_input = [dot(input_vector, weights[0][j]) for j in range(output_layer_size)]
    output_layer = [A * math.tanh(S * net_input[j]) for j in range(output_layer_size)]  # Compute the output of the first hidden layer for the first training image
    output_key = output_layer.index(max(output_layer))
    if output_key == y_test[c]:
        confusion_matrix[y_test[c]][y_test[c]] += 1
        correct += 1
    else:
        confusion_matrix[output_key][y_test[c]] += 1

sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu")
error = correct/len(x_test)
plt.show()


print("Test Dataset Accuracy: " + str(error))


data_folder = Path("Data")
data_folder.mkdir(exist_ok=True)  # Create the folder if it doesn't exist
weights_file_path = data_folder / "Net1_weights.txt"

active_weights = weights[0]

# Write the weights to the file
with open(weights_file_path, "w") as file:
    for neuron_weights in active_weights:
        # Convert each float weight to a string and join them with spaces
        line = " ".join(str(w) for w in neuron_weights)
        file.write(line + "\n")

print(f"Weights successfully saved to {weights_file_path}")