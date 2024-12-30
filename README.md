# Neural Network for MNIST Digit Recognition

This project implements a simple neural network from scratch to recognize handwritten digits from the MNIST dataset. The neural network is built using NumPy and demonstrates forward propagation, backpropagation, and gradient descent to train the model.

## Dataset

The dataset used is the MNIST digit dataset, containing grayscale images of size 28x28 pixels.

- **Train Data**: 60,000 examples
- **Test Data**: 10,000 examples
- Each image is flattened into a 784-dimensional vector (28x28).
- Labels range from 0 to 9, representing the digit in the image.

## Workflow

1. **Data Preprocessing**
   - Load the dataset as a CSV file.
   - Normalize pixel values to the range [0, 1].
   - Split the data into training and validation sets.

2. **Neural Network Architecture**
   - Input layer: 784 neurons (flattened image pixels).
   - Hidden layer: 10 neurons with ReLU activation.
   - Output layer: 10 neurons with Softmax activation.

3. **Functions Implemented**
   - **Initialization**: Randomly initialize weights and biases.
   - **Forward Propagation**:
     - Compute activations for hidden and output layers.
     - Use ReLU activation for the hidden layer and Softmax for the output layer.
   - **Backward Propagation**:
     - Compute gradients of weights and biases using the loss function.
     - Implement gradient descent to update weights and biases.
   - **Evaluation**:
     - Predict classes using the trained model.
     - Compute accuracy.

4. **Training**
   - Use the gradient descent algorithm.
   - Monitor accuracy at every 10 iterations.

## Key Functions

### Data Preprocessing
```python
np.random.shuffle(data)  # Shuffle dataset
X_train = X_train / 255.0  # Normalize features
```

### Neural Network Functions

#### Initialization
```python
def init_params():
    W1 = np.random.randn(10, 784) * np.sqrt(2/784)
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * np.sqrt(2/10)
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2
```

#### Activation Functions
```python
def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
```

#### Forward Propagation
```python
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
```

#### Backward Propagation
```python
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * (Z1 > 0)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2
```

#### Gradient Descent
```python
def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration {i}: Accuracy = {accuracy:.4f}")
    return W1, b1, W2, b2
```

## How to Run

1. Load the dataset into the `/kaggle/input/digit-recognizer/train.csv` path.
2. Execute the script to preprocess data, initialize parameters, and train the model.
3. Monitor the accuracy during training.

## Improvements
- Normalize the data for better convergence.
- Implement advanced weight initialization.
- Use more iterations and tune learning rate (`alpha`).

## Results
The accuracy should improve with more training iterations and optimized hyperparameters.
