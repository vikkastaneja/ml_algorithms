import pytest
import numpy as np
from LinearRegressionGD import LinearRegressionGD

# test_LinearRegressionGD.py


@pytest.fixture
def data():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    return X, y

def test_fit(data):
    X, y = data
    model = LinearRegressionGD()
    model.fit(X, y)
    assert model.weights is not None
    assert model.bias is not None

def test_predict(data):
    X, y = data
    model = LinearRegressionGD(n_iterations=5000)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == y.shape
    assert np.allclose(predictions, y, atol=1e-1)

def test_score(data):
    X, y = data
    model = LinearRegressionGD()
    model.fit(X, y)
    score = model.score(X, y)
    assert score > 0.9