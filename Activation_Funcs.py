import numpy as np

class Activation_ReLU: # Rectified Linear Activation Function (ReLU)
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax: # Softmax Activation Function
    # Forward propagation
    def forward(self, inputs):
        # The max of the inputs is substracted from the inputs to avoid overflow. e**exp would be overflow with exp=1000, but by substracting the max, exp <= 0 and the values will be between 0 and 1)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))     # keepdims=True to keep the shape of the array
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # Normalize the probabilities
        self.output = probabilities