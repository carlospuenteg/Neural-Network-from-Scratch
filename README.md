# Neural Network from scratch in Python

## Index
* [1. Introduction](#1-introduction)
* [2. Polynomial Class](#2-polynomial-class)
* [3. Create the sample dataset / input data](#3-create-the-sample-dataset---input-data)
* [4. Plot the dataset](#4-plot-the-dataset)
* [5. Create the Dense Layer](#5-create-the-dense-layer)
* [6. Rectified Linear Activation Function (ReLU)](#6-rectified-linear-activation-function---relu)
* [7. Softmax Activation Function](#7-softmax-activation-function)
* [8. Categorical Cross-Entropy Loss function](#8-categorical-cross-entropy-loss-function)
* [9. Prediction](#9-prediction)
* [10. Accuracy](#10-accuracy)
* [11. All together](#11-all-together)
* [12. Complete Neural Network](#12-complete-neural-network)
* [13. Full Script](#13-full-script)




## 1. Introduction

In this example, we will build a neural network that can identify a polynomial from a point in the plane.

For example, if there are two polynomials: 
```r
p1(x) = -x^2 + 1
p2(x) = x^3 - 2
```

<img src=readme-assets/pols.png width=500>

If the input is `(0, 1)`, the output would be `p1`, since `p1(0) = 1`.




## 2. Polynomial Class

```python
class P:
    # Create the polynomial with the given coefficients
    def __init__(self, *coeffs):
        self.coeffs = coeffs

    # Get the value of the polynomial with a given x
    def p(self, x):
        return sum(coef*(x**(len(self.coeffs)-1-exp)) for exp,coef in enumerate(self.coeffs))

    # Show the polynomial
    def __str__(self):
        toret = ""
        max_exp = len(self.coeffs) - 1
        for exp,coef in enumerate(self.coeffs):
            exp = max_exp - exp
            if coef != 0:
                var = "x" if exp != 0 else ""
                sp = " " if exp != max_exp else ""
                coef = f"{sp}{'-' if coef < 0 else '+' if exp != max_exp else ''}{sp}{abs(coef) if abs(coef) != 1 or exp == 0 else ''}"
                exp = f"^{exp}" if exp > 1 else ""
                toret += f"{coef}{var}{exp}"
        return toret
```

#### Example

```python
p1 = P(-1,0,1)
print(p1)       #>> -x^2 + 1
print(p1.p(2))  #>> -3
```




## 3. Create the sample dataset / input data

### Function

| Argument | Description | Example |
| -------- | ----------- | ------- |
| `n_samples` | The number of samples for each class. | `100` |
| `pols` | List of polynomial to generate the samples. | `[P(3,1), P(8,0,-2)]` |

```python
import numpy as np; np.random.seed(0)

# Polynomials to use in the neural network
pols = (P(3, 1), P(-2, -3), P(4, -1, -2), P(-1, 2, 0), P(3, 0, 2), P(1, -1, 3, 2), P(-3, 1, 0, 0), 
P(2, 0, 1, -2), P(5, -3, 1, -2, 2), P(-2, -3, 0, 4, 1), P(1, 0, -4, 2, 3), P(1, 0, 0, 0, -2))

# Generates a dataset with the chosen Polynomials
def generate_dataset(n_samples, pols=pols):
    X = np.zeros((n_samples*len(pols), 2))
    y = np.zeros(n_samples*len(pols), dtype='uint8')
    for i,pol in enumerate(pols):
        for j in range(n_samples):
            r = np.random.uniform(-1, 1)
            X[i*n_samples+j] = [r, pol.p(r)]
            y[i*n_samples+j] = i
    return X, y
```

### Example

```python
X, y = generate_dataset(100, pols)
```

#### Generated samples (`X`)
The integer labels for each sample is a matrix of `n_samples x n_features` (100x2).
- `n_samples` : 100
- `n_features` : 2 (x and y)

```python
[[ 0.77118056  3.31354169]
 [ 0.16839427  1.50518282]
 ...
 [-0.85168746 -1.47383613]
 [-0.22573398 -1.9974035 ]]
```

#### Integer labels for each sample (`y`)
The integer labels for each sample is a matrix of `n_samples x n_outputs` (100x1).
- `n_samples` : 100
- `n_outputs` : 1 (There are 12 different possible outputs, since there are 12 polynomials)
  
```python
[ 0  0  0 ... 11 11 11]
```




## 4. Plot the dataset

```python
def plot_dataset(X, y, title='Dataset'):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10')
    plt.show()
```

#### Result

<img src=readme-assets/dataset.png width=500>




## 5. Create the Dense Layer

### What is a Dense Layer?
A dense layer is a layer that is deeply connected with its preceding layer (each neuron receives input from all the neurons in the preceding layer).

<img src=readme-assets/neural-network.png width=500>

### Class

#### `__init__` (Initialize the weights and bias)
| Argument | Description | Example |
| -------- | ----------- | ------- |
| n_inputs | Number of inputs. | `2` (x and y) |
| n_neurons | Number of neurons used | `12` (n_neurons >= n_outputs) |

#### `forward` (Forward propagation)
| Argument | Description | Example |
| -------- | ----------- | ------- |
| inputs | The inputs to the layer. | `X`

```python
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
```

### Example

```python
# Create a dense layer with 2 inputs and 12 neurons
layer1 = Dense_Layer(n_inputs=2, n_neurons=12)

# Generate the output of the layer by multiplying the inputs by the weights and adding the biases
layer1.forward(X)
```

#### Layer 1 weights
The random weights are a matrix of `n_inputs x n_neurons` (2x12).
```python
[[ 0.03528166 -0.01527744 -0.12986867  0.12760753  0.13250141  0.02053326
   0.0045134   0.23396248 -0.02764328 -0.0259577   0.03644812  0.1471322 ]
 [ 0.15927708 -0.02585726  0.03083312 -0.13780835 -0.03119761 -0.08402904
  -0.10068318  0.16815767 -0.07922867 -0.05316059  0.03658488  0.12978253]]
```

#### Layer 1 biases
The biases are a matrix of `1 x n_neurons` (1x12).
```python
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
```

#### Layer 1 output
The output of the layer is a matrix of `n_samples x n_neurons` (100x12).
```python
[[ 0.20937075 -0.03492186  0.02718487 ... -0.07126449  0.05085822 0.18215744]
 [ 0.38010995 -0.0658176   0.01475017 ... -0.1329698   0.09950744 0.36067201]
 ...
 [-0.32540249  0.0547511  -0.03543965 ...  0.11146536 -0.08045569 -0.28900613]]
```




# 6. Rectified Linear Activation Function (ReLU)

The ReLU is used to transform the outputs of the previous layer into a non-negative output.

It leaves the positive inputs unchanged and transforms the negative inputs into zeros.

### Arguments

#### `forward` (Forward propagation)
| Argument | Description | Example |
| -------- | ----------- | ------- |
| `inputs` | The inputs to be transformed. | `layer1.output`

### Class

```python
class Activation_ReLU:
    # Forward propagation
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
```

### Example

```python
# Create a ReLU activation function
act1 = Activation_ReLU()

# Change the layer1 output with the activation function
act1.forward(layer1.output)
```

#### Layer 1 output (before ReLU)
The output of the layer is a matrix of `n_samples x n_neurons` (100x12).
```python
[[ 0.20937075 -0.03492186  0.02718487 ... -0.07126449  0.05085822 0.18215744]
 [ 0.38010995 -0.0658176   0.01475017 ... -0.1329698   0.09950744 0.36067201]
 ...
 [-0.32540249  0.0547511  -0.03543965 ...  0.11146536 -0.08045569 -0.28900613]]
```

#### Layer 1 output (after ReLU)
The output of the layer is a matrix of `n_samples x n_neurons` (100x12).
```python
[[0.20937075 0.         0.02718487 ... 0.         0.05085822 0.18215744]
 [0.38010995 0.         0.01475017 ... 0.         0.09950744 0.36067201]
 ...
 [0.         0.0547511  0.         ... 0.11146536 0.         0.        ]]
```




## 7. Softmax Activation Function

The Softmax activation function is used to transform the outputs of the previous layer into a probability distribution.

### Arguments

#### `forward` (Forward propagation)
| Argument | Description | Example |
| -------- | ----------- | ------- |
| `inputs` | The inputs to be transformed. | `layer2.output`


### Class

```python
class Activation_Softmax: # Softmax Activation Function
    # Forward propagation
    def forward(self, inputs):
        # The max of the inputs is substracted from the inputs to avoid overflow. e**exp would be overflow with exp=1000, but by substracting the max, exp <= 0 and the values will be between 0 and 1)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))     # keepdims=True to keep the shape of the array
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # Normalize the probabilities
        self.output = probabilities
```

### Example

```python
# Create a new Dense Layer with 12 inputs (the ReLU output) and 12 neurons
layer2 = layer(n_inputs=12, n_neurons=12)

# Generate the outputs using the act1.output as inputs
layer2.forward(act1.output)

# Create a Softmax activation function
act2 = Activation_Softmax()

# Change the layer2 output with the activation function
act2.forward(layer2.output)
```

#### Layer 2 weights
The random weights are a matrix of `n_inputs x n_neurons` (12x12).
```python
[[ 0.04811151  0.27593551 -0.0074668   0.02587164  0.02756007  0.14350494
   0.0507239  -0.01162297 -0.09474886  0.02444435  0.14013448 -0.04103818]
 ...
 [ 0.00875315  0.09387469  0.06071117 -0.10481704 -0.08602625  0.03283013
  -0.04012978 -0.03166553  0.05969065 -0.09872867 -0.04012347 -0.08000825]]
```

#### Layer 2 biases
The biases are a matrix of `1 x n_neurons` (1x12).
```python
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
```

#### Layer 2 output (before Softmax)
The output of the layer is a matrix of `n_samples x n_neurons` (100x12).
```python
[[-0.01228636  0.0526258   0.03871606 ...  0.00462124  0.04428541 -0.01437666]
 [-0.0271324   0.08745892  0.07535303 ...  0.01077147  0.08546539 -0.02271744]
 ...
 [-0.023293   -0.06307724  0.04470089 ...  0.00960348  0.07367412 -0.02288441]]
```

#### Layer 2 output (after Softmax)
The output of the layer is a matrix of `n_samples x n_neurons` (100x12).
```python
[[0.08163237 0.08710707 0.08590382 ... 0.08302431 0.08638358 0.08146191]
 [0.07999438 0.0897069  0.08862746 ... 0.08308467 0.08952824 0.08034833]
 ...
 [0.08068105 0.07753423 0.08635767 ... 0.08337931 0.08889633 0.08071402]]
```




## 8. Categorical Cross-Entropy Loss function

The Categorical Cross-Entropy Loss function is used to calculate the loss between the predicted probabilities and the true labels.

### Arguments

#### `forward` (Forward propagation)
| Argument | Description | Example |
| -------- | ----------- | ------- |
| `y_pred` | The calculated predictions. | `act2.output` |
| `y_true` | The true labels. | `y` |


### Class

```python
class Loss: # Loss function
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss): # Inherits from Loss class
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) # To avoid log(0)

        if len(y_true.shape) == 1: # If the passed values are scalar. 1D array with the correct classes
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: # If the passed values are one-hot encoded vectors. 2D array with vectors containing 0s and a 1, the 1 being the correct (one-hot encoding).
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
```

### Example

```python
# Create a Loss_CategoricalCrossentropy object
loss_function = Loss_CategoricalCrossentropy()

# Calculate the loss with the loss_function.calculate method
loss = loss_function.calculate(act2.output, y)
```

#### Loss output
```python
2.4942046014713335
```




## 9. Prediction

Polynomial which is the most likely to be the correct one for each input

```python
predictions = np.argmax(act2.output, axis=1)
```
```python
[ 1  1  1 ... 10  4 10]
```



## 10. Accuracy

Accuracy of the previous prediction.

It will be low since the model has not been trained yet.

```python
accuracy = np.mean(predictions == y)
```
```
0.015833333333333335
```




## 11. All together

```python
pols = (P(3, 1), P(-2, -3), P(4, -1, -2), P(-1, 2, 0), P(3, 0, 2), P(1, -1, 3, 2), P(-3, 1, 0, 0), 
P(2, 0, 1, -2), P(5, -3, 1, -2, 2), P(-2, -3, 0, 4, 1), P(1, 0, -4, 2, 3), P(1, 0, 0, 0, -2))

n_classes = len(pols)

X, y = generate_dataset(100, pols)

layer1 = Dense_Layer(n_inputs= 2, n_neurons = n_classes)
layer1.forward(X)

act1 = Activation_ReLU()
act1.forward(layer1.output)

layer2 = Dense_Layer(n_inputs=12, n_neurons=12)
layer2.forward(act1.output)

act2 = Activation_Softmax()
act2.forward(layer2.output)

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(act2.output, y)

predictions = np.argmax(act2.output, axis=1)

accuracy = np.mean(predictions == y)
```




## 12. Complete Neural Network 

### Imports

```python
import numpy as np;
import matplotlib.pyplot as plt
import os
from Activation_Funcs import *
from Dense_Layer import *
from Loss import *
from Polynomial import P
```

### Polynomials to be used

```python
pols = (P(3, 1), P(-2, -3), P(4, -1, -2), P(-1, 2, 0), P(3, 0, 2), P(1, -1, 3, 2), P(-3, 1, 0, 0), 
P(2, 0, 1, -2), P(5, -3, 1, -2, 2), P(-2, -3, 0, 4, 1), P(1, 0, -4, 2, 3), P(1, 0, 0, 0, -2))
```

#### Plot the polynomials

```python
def plot_polynomials(pols, ran=(-2,2), n_samples=100):
    # 100 linearly spaced numbers
    x = np.linspace(ran[0],ran[1],n_samples)

    # setting the axes at the centre
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # plot the functions
    for pol in pols:
        y = pol.p(x)
        plt.plot(x, y, label=str(pol))

    plt.legend(loc='upper left')

    # show the plot
    plt.show()
```

```python
# Example
plot_polynomials(
    pols=(P(1,0,1), P(-2, 3)),
    ran=(-1,1),
    n_samples=100
)
```

<img src=readme-assets/pols-plot.png width=400>


### Train the neural network

```python
def train(n_samples=100, n_epochs=100000, learn_rate=0.0005, pols=pols):
    n_classes = len(pols)

    X, y = generate_dataset(n_samples, pols)
    # plot_dataset(X, y)

    layer1 = Dense_Layer(n_inputs= 2,        n_neurons = n_classes)
    layer2 = Dense_Layer(n_inputs=n_classes, n_neurons = n_classes)
    act1 = Activation_ReLU()
    act2 = Activation_Softmax()
    loss_func = Loss_CategoricalCrossentropy()

    lowest_loss = np.inf
    b_weights_l1 = layer1.weights.copy()
    b_biases_l1  = layer1.biases.copy()
    b_weights_l2 = layer2.weights.copy()
    b_biases_l2  = layer2.biases.copy()

    for iter in range(n_epochs):
        # Randomize weights and biases
        layer1.weights +=   learn_rate * np.random.randn(layer1.weights.shape[0], layer1.weights.shape[1])
        layer1.biases +=    learn_rate * np.random.randn(layer1.biases.shape[0], layer1.biases.shape[1])
        layer2.weights +=   learn_rate * np.random.randn(layer2.weights.shape[0], layer2.weights.shape[1])
        layer2.biases +=    learn_rate * np.random.randn(layer2.biases.shape[0], layer2.biases.shape[1])

        # Forward propagation
        layer1.forward(X)
        act1.forward(layer1.output)
        layer2.forward(act1.output)
        act2.forward(layer2.output)

        # Calculate loss
        loss = loss_func.calculate(act2.output, y)

        # Results. If [0 0 1 0 0] it will return 2. argmax returns the index of the maximum value in each array
        predictions = np.argmax(act2.output, axis=1)

        # Accuracy of the results compared to the true values
        accuracy = np.mean(predictions == y)
        
        if loss < lowest_loss:
            print(f"New set of weights and biases found, iter: {iter}, loss: {loss}, acc: {accuracy}")
            lowest_loss = loss
            b_weights_l1 = layer1.weights.copy()
            b_biases_l1 = layer1.biases.copy()
            b_weights_l2 = layer2.weights.copy()
            b_biases_l2 = layer2.biases.copy()
        else:
            layer1.weights = b_weights_l1.copy()
            layer1.biases = b_biases_l1.copy()
            layer2.weights = b_weights_l2.copy()
            layer2.biases = b_biases_l2.copy()

    save_best(layer1, layer2, loss, accuracy)
```

#### Example

```python
train(
    n_samples=100, 
    n_epochs=1000000, 
    learn_rate=0.0005,
    pols=pols
)
```

```bash
# Training...
New set of weights and biases found, iter: 0, loss: 2.4899427650572044, acc: 0.11333333333333333
New set of weights and biases found, iter: 3, loss: 2.489925294343727, acc: 0.11083333333333334
New set of weights and biases found, iter: 5, loss: 2.4899040051678587, acc: 0.0975
New set of weights and biases found, iter: 6, loss: 2.4897983233720917, acc: 0.10833333333333334
...
New set of weights and biases found, iter: 998633, loss: 0.8934836266737981, acc: 0.64
New set of weights and biases found, iter: 998652, loss: 0.8934751585906433, acc: 0.6416666666666667
New set of weights and biases found, iter: 998792, loss: 0.8934622116619763, acc: 0.6391666666666667
New set of weights and biases found, iter: 998841, loss: 0.8934603025209412, acc: 0.6416666666666667
...
New best loss: 0.6669996962075785, accuracy: 77.83%
```

#### Save the best weights and biases

```python
def save_best(layer1, layer2, loss, accuracy):
    if not os.path.exists('w&b'):
        os.makedirs('w&b')
    if not os.path.exists('w&b/loss.npy') or loss < np.load('w&b/loss.npy'):
        np.save('w&b/weights_l1.npy', layer1.weights)
        np.save('w&b/biases_l1.npy', layer1.biases)
        np.save('w&b/weights_l2.npy', layer2.weights)
        np.save('w&b/biases_l2.npy', layer2.biases)
        np.save('w&b/loss.npy', loss)
        np.save('w&b/accuracy.npy', accuracy)
        print(f"\nNew best loss: {loss}, accuracy: {accuracy*100:.2f}%")
```


### Test the neural network

```python
def test(x,y):
    input = [x, y]
    layer1 = Dense_Layer(n_inputs = 2,  n_neurons = 10)
    layer2 = Dense_Layer(n_inputs = 10, n_neurons = 10)
    act1 = Activation_ReLU()
    act2 = Activation_Softmax()

    layer1.weights = np.load('w&b/weights_l1.npy')
    layer1.biases  = np.load('w&b/biases_l1.npy')
    layer2.weights = np.load('w&b/weights_l2.npy')
    layer2.biases  = np.load('w&b/biases_l2.npy')
    loss = np.load('w&b/loss.npy')

    layer1.forward(input)
    act1.forward(layer1.output)
    layer2.forward(act1.output)
    act2.forward(layer2.output)

    print(f"Overall loss: {loss}")

    predictions = act2.output[0].argsort()[::-1]
    print(f"Sorted predictions: {predictions}")

    prediction = predictions[0]
    odds = act2.output[0][prediction]
    return f"\nPolynomial {prediction} ({odds*100:.2f}%)\n"
```

#### Example

```python
print(
    test(-0.55, pols[9].p(-0.55))
)
```

```bash
Overall loss: 0.6669996962075785
Overall accuracy: 77.83%
Sorted predictions: [ 9  2  0 10 11  3  5  1  4  7  6  8]

Polynomial 9 (65.92%)
```



## 13. Full Script

Link to the script -> [neural-net.py](neural-net.py)

```python
import numpy as np;
import matplotlib.pyplot as plt
import os
from Activation_Funcs import *
from Dense_Layer import *
from Loss import *
from Polynomial import P

#------------------------------------------------------------------------------
# Polynomials
pols = (P(3, 1), P(-2, -3), P(4, -1, -2), P(-1, 2, 0), P(3, 0, 2), P(1, -1, 3, 2), P(-3, 1, 0, 0), 
P(2, 0, 1, -2), P(5, -3, 1, -2, 2), P(-2, -3, 0, 4, 1), P(1, 0, -4, 2, 3), P(1, 0, 0, 0, -2))

#------------------------------------------------------------------------------
# Generates a dataset with the chosen Polynomials
def generate_dataset(n_samples, pols=pols):
    X = np.zeros((n_samples*len(pols), 2))
    y = np.zeros(n_samples*len(pols), dtype='uint8')
    for i,pol in enumerate(pols):
        for j in range(n_samples):
            r = np.random.uniform(-1, 1)
            X[i*n_samples+j] = [r, pol.p(r)]
            y[i*n_samples+j] = i
    return X, y

#------------------------------------------------------------------------------

def plot_dataset(X, y, title='Dataset'):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10')
    plt.show()

#------------------------------------------------------------------------------

def plot_polynomials(pols, ran=(-2,2), n_samples=100):
    # 100 linearly spaced numbers
    x = np.linspace(ran[0],ran[1],n_samples)

    # setting the axes at the centre
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # plot the functions
    for pol in pols:
        y = pol.p(x)
        plt.plot(x, y, label=str(pol))

    plt.legend(loc='upper left')

    # show the plot
    plt.show()

#-----------------------------------------------------------------------------------

def save_best(layer1, layer2, loss, accuracy):
    if not os.path.exists('w&b'):
        os.makedirs('w&b')
    if not os.path.exists('w&b/loss.npy') or loss < np.load('w&b/loss.npy'):
        np.save('w&b/weights_l1.npy', layer1.weights)
        np.save('w&b/biases_l1.npy', layer1.biases)
        np.save('w&b/weights_l2.npy', layer2.weights)
        np.save('w&b/biases_l2.npy', layer2.biases)
        np.save('w&b/loss.npy', loss)
        np.save('w&b/accuracy.npy', accuracy)
        print(f"\nNew best loss: {loss}, accuracy: {accuracy*100:.2f}%")

#-----------------------------------------------------------------------------------

def train(n_samples=100, n_epochs=100000, learn_rate=0.0005, pols=pols):
    n_classes = len(pols)

    X, y = generate_dataset(n_samples, pols)
    # plot_dataset(X, y)

    layer1 = Dense_Layer(n_inputs= 2,        n_neurons = n_classes)
    layer2 = Dense_Layer(n_inputs=n_classes, n_neurons = n_classes)
    act1 = Activation_ReLU()
    act2 = Activation_Softmax()
    loss_func = Loss_CategoricalCrossentropy()

    lowest_loss = np.inf
    b_weights_l1 = layer1.weights.copy()
    b_biases_l1  = layer1.biases.copy()
    b_weights_l2 = layer2.weights.copy()
    b_biases_l2  = layer2.biases.copy()

    for iter in range(n_epochs):
        # Randomize weights and biases
        layer1.weights +=   learn_rate * np.random.randn(layer1.weights.shape[0], layer1.weights.shape[1])
        layer1.biases +=    learn_rate * np.random.randn(layer1.biases.shape[0], layer1.biases.shape[1])
        layer2.weights +=   learn_rate * np.random.randn(layer2.weights.shape[0], layer2.weights.shape[1])
        layer2.biases +=    learn_rate * np.random.randn(layer2.biases.shape[0], layer2.biases.shape[1])

        # Forward propagation
        layer1.forward(X)
        act1.forward(layer1.output)
        layer2.forward(act1.output)
        act2.forward(layer2.output)

        # Calculate loss
        loss = loss_func.calculate(act2.output, y)

        # Results. If [0 0 1 0 0] it will return 2. argmax returns the index of the maximum value in each array
        predictions = np.argmax(act2.output, axis=1)

        # Accuracy of the results compared to the true values
        accuracy = np.mean(predictions == y)
        
        if loss < lowest_loss:
            print(f"New set of weights and biases found, iter: {iter}, loss: {loss}, acc: {accuracy}")
            lowest_loss = loss
            b_weights_l1 = layer1.weights.copy()
            b_biases_l1 = layer1.biases.copy()
            b_weights_l2 = layer2.weights.copy()
            b_biases_l2 = layer2.biases.copy()
        else:
            layer1.weights = b_weights_l1.copy()
            layer1.biases = b_biases_l1.copy()
            layer2.weights = b_weights_l2.copy()
            layer2.biases = b_biases_l2.copy()

    save_best(layer1, layer2, loss, accuracy)

#-----------------------------------------------------------------------------------

# Check if the network works
def test(x,y):
    input = [x, y]
    layer1 = Dense_Layer(n_inputs = 2,  n_neurons = 10)
    layer2 = Dense_Layer(n_inputs = 10, n_neurons = 10)
    act1 = Activation_ReLU()
    act2 = Activation_Softmax()

    layer1.weights = np.load('w&b/weights_l1.npy')
    layer1.biases  = np.load('w&b/biases_l1.npy')
    layer2.weights = np.load('w&b/weights_l2.npy')
    layer2.biases  = np.load('w&b/biases_l2.npy')
    loss = np.load('w&b/loss.npy')
    accuracy = np.load('w&b/accuracy.npy')

    layer1.forward(input)
    act1.forward(layer1.output)
    layer2.forward(act1.output)
    act2.forward(layer2.output)

    print(f"Overall loss: {loss}")
    print(f"Overall accuracy: {accuracy*100:.2f}%")

    predictions = act2.output[0].argsort()[::-1]
    print(f"Sorted predictions: {predictions}")

    prediction = predictions[0]
    odds = act2.output[0][prediction]
    return f"\nPolynomial {prediction} ({odds*100:.2f}%)\n"

#-----------------------------------------------------------------------------------
"""
train(
    n_samples=100, 
    n_epochs=10000, 
    learn_rate=0.0005,
    pols=pols
)
"""
print(
    test(-0.55, pols[9].p(-0.55))
)
```