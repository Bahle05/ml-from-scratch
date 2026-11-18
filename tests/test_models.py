import numpy as np
import pytest

from mlscratch.linear_regression import (LinearRegressionGD, StandardScaler, add_bias,
                                         compute_cost, normal_equation, r2_score)
from mlscratch.classifiers import (LogisticRegressionGD, NeuralNetwork, binary_cross_entropy,
                                   classification_report, sigmoid)


def test_cost_is_zero_for_perfect_fit():
    X = add_bias(np.array([[1.0], [2.0], [3.0]]))
    y = np.array([3.0, 5.0, 7.0])                 # y = 1 + 2x
    assert compute_cost(X, y, np.array([1.0, 2.0])) == pytest.approx(0.0)


def test_gradient_descent_matches_normal_equation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = 4 + X @ np.array([1.5, -2.0, 0.5]) + rng.normal(scale=0.1, size=200)
    model = LinearRegressionGD(learning_rate=0.1, iterations=3000).fit(X, y)
    np.testing.assert_allclose(model.theta_, normal_equation(add_bias(X), y), atol=1e-6)
    assert r2_score(y, model.predict(X)) > 0.99


def test_cost_decreases_every_iteration():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 2))
    y = X @ np.array([3.0, -1.0])
    history = LinearRegressionGD(learning_rate=0.05, iterations=200).fit(X, y).cost_history_
    assert all(b <= a for a, b in zip(history, history[1:]))


def test_scaler_handles_constant_feature():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    Xs = StandardScaler().fit_transform(X)
    assert np.all(np.isfinite(Xs))
    assert Xs[:, 0].mean() == pytest.approx(0.0)


def test_sigmoid_is_stable_for_extreme_inputs():
    out = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.all(np.isfinite(out))
    assert out[1] == pytest.approx(0.5)


def test_cross_entropy_never_infinite():
    assert np.isfinite(binary_cross_entropy(np.array([1, 0]), np.array([0.0, 1.0])))


def test_logistic_regression_separates_linear_data():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegressionGD(learning_rate=0.5, iterations=1000).fit(X, y)
    assert (model.predict(X) == y).mean() > 0.97


def test_neural_network_learns_xor():
    """XOR is not linearly separable, so this checks the hidden layer and backprop work."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 25, dtype=float)
    y = np.array([0, 1, 1, 0] * 25)
    nn = NeuralNetwork(hidden_units=4, learning_rate=1.0, iterations=2000, seed=1).fit(X * 2 - 1, y)
    assert (nn.predict(X * 2 - 1) == y).all()


def test_classification_report_counts():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])
    r = classification_report(y_true, y_pred)
    assert r["confusion"] == {"tp": 2, "tn": 1, "fp": 1, "fn": 1}
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(2 / 3)
