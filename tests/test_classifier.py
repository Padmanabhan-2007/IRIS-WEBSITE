import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_and_prepare_data
from src.generative_classifier import GaussianGenerativeClassifier

@pytest.fixture
def data():
    X_train, X_test, y_train, y_test, feature_names, target_names = load_and_prepare_data()
    return X_train, X_test, y_train, y_test, feature_names, target_names

@pytest.fixture
def trained_model(data):
    X_train, _, y_train, _, _, _ = data
    model = GaussianGenerativeClassifier()
    model.fit(X_train, y_train)
    return model

def test_classifier_can_fit_iris(trained_model):
    """Test 1: Classifier can fit the Iris dataset."""
    assert trained_model.classes_ is not None
    assert len(trained_model.classes_) > 0

def test_three_classes(trained_model):
    """Test 2: There are exactly 3 classes."""
    assert len(trained_model.classes_) == 3
    assert set(trained_model.classes_) == {0, 1, 2}

def test_mean_vector_shape(trained_model):
    """Test 3: Each mean vector has shape (4,)."""
    for c in trained_model.classes_:
        assert trained_model.means_[c].shape == (4,)

def test_covariance_matrix_shape(trained_model):
    """Test 4: Each covariance matrix has shape (4,4)."""
    for c in trained_model.classes_:
        assert trained_model.covariances_[c].shape == (4, 4)

def test_covariance_matrices_symmetric(trained_model):
    """Test 5: Covariance matrices are symmetric."""
    for c in trained_model.classes_:
        cov = trained_model.covariances_[c]
        np.testing.assert_array_almost_equal(cov, cov.T)

def test_class_priors_sum_to_1(trained_model):
    """Test 6: Class priors sum to 1."""
    priors_sum = sum(trained_model.priors_.values())
    assert np.isclose(priors_sum, 1.0)

def test_posterior_probabilities_sum_to_1(trained_model, data):
    """Test 7: Posterior probabilities sum to approximately 1."""
    _, X_test, _, _, _, _ = data
    posteriors = trained_model.predict_proba(X_test)
    sums = np.sum(posteriors, axis=1)
    np.testing.assert_array_almost_equal(sums, np.ones_like(sums), decimal=5)

def test_prediction_belongs_to_valid_classes(trained_model, data):
    """Test 8: Prediction belongs to one of setosa, versicolor, virginica (0, 1, 2)."""
    _, X_test, _, _, _, _ = data
    preds = trained_model.predict(X_test)
    for p in preds:
        assert p in [0, 1, 2]

def test_likelihood_values_finite(trained_model, data):
    """Test 9: Likelihood values are finite."""
    _, X_test, _, _, _, _ = data
    for c in trained_model.classes_:
        log_lik = trained_model._log_gaussian_density(X_test, trained_model.means_[c], trained_model.covariances_[c])
        assert np.all(np.isfinite(log_lik))

def test_model_does_not_crash_on_unknown_input(trained_model):
    """Test 10: The model does not crash on valid unknown flower input."""
    x_unknown = np.array([6.0, 2.9, 4.5, 1.5])
    pred = trained_model.predict(x_unknown.reshape(1, -1))
    assert pred[0] in [0, 1, 2]

def test_argmax_posterior_agrees_with_discriminant_score(trained_model, data):
    """Test 11: Predicted class from argmax posterior agrees with argmax discriminant score."""
    _, X_test, _, _, _, _ = data
    posteriors = trained_model.predict_proba(X_test)
    pred_posterior = np.argmax(posteriors, axis=1)
    
    scores = trained_model.decision_function(X_test)
    pred_score = np.argmax(scores, axis=1)
    
    np.testing.assert_array_equal(pred_posterior, pred_score)
