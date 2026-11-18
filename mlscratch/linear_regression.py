"""Multivariate linear regression implemented from scratch with NumPy.

The model predicts y from features X using h(x) = theta_0 + theta_1*x_1 + ... + theta_n*x_n.
Parameters are learned by minimising the mean squared error cost with batch gradient descent.
A closed-form normal-equation solver is included to check that gradient descent converges
to the same answer.
"""

import numpy as np


def add_bias(X):
    """Prepend a column of ones so theta_0 acts as the intercept."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def compute_cost(X, y, theta):
    """Mean squared error cost J(theta) = (1 / 2m) * sum((X @ theta - y)^2)."""
    m = len(y)
    errors = X @ theta - y
    return float(errors @ errors) / (2 * m)


def gradient_descent(X, y, theta, learning_rate, iterations):
    """Batch gradient descent. Returns the learned theta and the cost at every iteration."""
    m = len(y)
    history = []
    for _ in range(iterations):
        gradient = X.T @ (X @ theta - y) / m
        theta = theta - learning_rate * gradient
        history.append(compute_cost(X, y, theta))
    return theta, history


def normal_equation(X, y):
    """Closed-form least-squares solution theta = (X^T X)^-1 X^T y (pinv for numerical safety)."""
    return np.linalg.pinv(X.T @ X) @ X.T @ y


class StandardScaler:
    """Feature scaling to zero mean and unit variance, fitted on training data only."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # avoid dividing by zero for constant features
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LinearRegressionGD:
    """Linear regression trained with gradient descent, with a scikit-learn style interface."""

    def __init__(self, learning_rate=0.1, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, y):
        Xb = add_bias(X)
        theta0 = np.zeros(Xb.shape[1])
        self.theta_, self.cost_history_ = gradient_descent(
            Xb, y, theta0, self.learning_rate, self.iterations
        )
        return self

    def predict(self, X):
        return add_bias(X) @ self.theta_


def r2_score(y_true, y_pred):
    """Coefficient of determination: 1 - SS_res / SS_tot."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
