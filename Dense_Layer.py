import numpy as np

class Dense_Layer:
    # Initialize the weights and bias
    def __init__(self, n_inputs, n_neurons):
        # Random (with standard normal distribution) initial weights. Multiplied by 0.1 to have small values
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)

        # Initial biases with a value of 0
        self.biases = np.zeros((1, n_neurons))

    # Forward propagation (generate the outputs)
    def forward(self, inputs):
        # Multiply the inputs by the weights and add the biases
        self.output = np.dot(inputs, self.weights) + self.biases