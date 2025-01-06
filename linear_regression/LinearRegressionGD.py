import numpy as np
from sklearn.metrics import mean_squared_error

"""
LinearRegressionGD class implements a linear regression model using gradient descent optimization.

Gradient Descent for Linear Regression:
Gradient descent is an optimization algorithm used to minimize the cost function in linear regression. The cost function measures the error between the predicted values and the actual values. The goal is to find the optimal weights and bias that minimize this error.

The steps involved in gradient descent for linear regression are as follows:
1. Initialize the weights and bias to zero or small random values.
2. Calculate the predicted values using the current weights and bias.
3. Compute the cost function, which is the mean squared error between the predicted values and the actual values.
4. Calculate the gradients of the cost function with respect to the weights and bias.
5. Update the weights and bias by subtracting the learning rate times the gradients.
6. Repeat steps 2-5 for a specified number of iterations or until the cost function converges.

Attributes:
    learning_rate (float): The learning rate for gradient descent.
    n_iterations (int): The number of iterations for gradient descent.
    weights (np.ndarray): The weights of the linear regression model.
    bias (float): The bias of the linear regression model.
"""
class LinearRegressionGD:
    """
    Initialize the LinearRegressionGD model with the specified learning rate and number of iterations.

    Args:
        learning_rate (float): The learning rate for gradient descent. Default is 0.01.
        n_iterations (int): The number of iterations for gradient descent. Default is 1000.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        # Initialize the learning rate and number of iterations
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        # Initialize weights and bias to None, they will be set during fitting
        self.weights = None
        self.bias = None

    """
    Fit the linear regression model to the training data using gradient descent.

    Args:
        X (np.ndarray): The input data of shape (n_samples, n_features).
        y (np.ndarray): The target values of shape (n_samples,).

    Returns:
        None
    """
    def fit(self, X, y):
        # Get the number of samples (n_samples) and number of features (n_features) from the input data X
        n_samples, n_features = X.shape

        # Initialize weights as a zero vector of shape (n_features,) and bias as zero
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Perform gradient descent for a specified number of iterations
        for _ in range(self.n_iterations):
            # Calculate the predicted values using the current weights and bias
            y_predicted = np.dot(X, self.weights) + self.bias

            # Calculate the gradient of the loss function with respect to weights
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))

            # Calculate the gradient of the loss function with respect to bias
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update the weights by subtracting the learning rate times the gradient
            self.weights -= self.learning_rate * dw

            # Update the bias by subtracting the learning rate times the gradient
            self.bias -= self.learning_rate * db

    """
    Predict the target values for the given input data using the learned weights and bias.

    Args:
        X (np.ndarray): The input data of shape (n_samples, n_features).

    Returns:
        np.ndarray: The predicted target values of shape (n_samples,).
    """
    def predict(self, X):
        # Predict the target values using the learned weights and bias
        return np.dot(X, self.weights) + self.bias

    """
    Compute the mean squared error of the model on the given data.

    Args:
        X (np.ndarray): The input data of shape (n_samples, n_features).
        y (np.ndarray): The true target values of shape (n_samples,).

    Returns:
        float: The mean squared error of the model.
    """
    def score(self, X, y):
        y_pred = self.predict(X)
        return 1 - mean_squared_error(y, y_pred)