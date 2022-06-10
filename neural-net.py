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