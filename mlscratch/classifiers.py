"""Binary classifiers implemented from scratch with NumPy.

- LogisticRegressionGD: a linear decision boundary trained with gradient descent on the
  cross-entropy loss, with optional L2 regularisation.
- NeuralNetwork: one hidden layer (tanh) and a sigmoid output, trained with
  backpropagation. It can learn non-linear decision boundaries that logistic regression cannot.
"""

import numpy as np


def sigmoid(z):
    # Clip to keep np.exp from overflowing on very large negative inputs
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def binary_cross_entropy(y, p, eps=1e-12):
    """Average log loss. eps keeps log() away from zero."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class LogisticRegressionGD:
    def __init__(self, learning_rate=0.1, iterations=2000, l2=0.0):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2 = l2

    def fit(self, X, y):
        m, n = X.shape
        self.w_ = np.zeros(n)
        self.b_ = 0.0
        self.loss_history_ = []
        for _ in range(self.iterations):
            p = sigmoid(X @ self.w_ + self.b_)
            error = p - y
            # Gradients of the cross-entropy loss (bias is not regularised)
            dw = X.T @ error / m + (self.l2 / m) * self.w_
            db = error.mean()
            self.w_ -= self.learning_rate * dw
            self.b_ -= self.learning_rate * db
            loss = binary_cross_entropy(y, p) + (self.l2 / (2 * m)) * float(self.w_ @ self.w_)
            self.loss_history_.append(loss)
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w_ + self.b_)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class NeuralNetwork:
    """Input -> hidden layer (tanh) -> output (sigmoid), trained with full-batch backpropagation."""

    def __init__(self, hidden_units=8, learning_rate=0.5, iterations=3000, seed=42):
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.seed = seed

    def _init_params(self, n_features):
        rng = np.random.default_rng(self.seed)
        # Xavier initialisation: random weights break symmetry so hidden units learn different
        # features, and scaling by 1/sqrt(fan_in) keeps tanh out of its flat regions. Very small
        # weights (e.g. 0.01) leave the network stuck near a saddle point for thousands of steps.
        self.W1_ = rng.normal(0, 1 / np.sqrt(n_features), size=(n_features, self.hidden_units))
        self.b1_ = np.zeros(self.hidden_units)
        self.W2_ = rng.normal(0, 1 / np.sqrt(self.hidden_units), size=(self.hidden_units, 1))
        self.b2_ = np.zeros(1)

    def _forward(self, X):
        Z1 = X @ self.W1_ + self.b1_
        A1 = np.tanh(Z1)
        Z2 = A1 @ self.W2_ + self.b2_
        A2 = sigmoid(Z2).ravel()
        return A1, A2

    def fit(self, X, y):
        m = X.shape[0]
        self._init_params(X.shape[1])
        self.loss_history_ = []
        for _ in range(self.iterations):
            A1, A2 = self._forward(X)
            self.loss_history_.append(binary_cross_entropy(y, A2))

            # Backpropagation
            dZ2 = (A2 - y).reshape(-1, 1) / m          # output layer error
            dW2 = A1.T @ dZ2
            db2 = dZ2.sum(axis=0)
            dZ1 = (dZ2 @ self.W2_.T) * (1 - A1 ** 2)    # tanh'(z) = 1 - tanh(z)^2
            dW1 = X.T @ dZ1
            db1 = dZ1.sum(axis=0)

            self.W1_ -= self.learning_rate * dW1
            self.b1_ -= self.learning_rate * db1
            self.W2_ -= self.learning_rate * dW2
            self.b2_ -= self.learning_rate * db2
        return self

    def predict_proba(self, X):
        return self._forward(X)[1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def classification_report(y_true, y_pred):
    """Accuracy, precision, recall, F1 and the confusion matrix counts."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }
