import numpy as np

class Loss: # Loss function
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss): # Inherits from Loss class
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) # To avoid log(0)

        if len(y_true.shape) == 1: # They have passed scalar values. 1D array with the correct classes
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: # They have passed one-hot encoded vectors. 2D array with vectors containing 0s and a 1, the 1 being the correct (one-hot encoding).
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods