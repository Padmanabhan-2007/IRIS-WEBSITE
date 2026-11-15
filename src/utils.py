# Utility functions if any are needed
import numpy as np

def explain_prediction(x, model, target_names):
    """
    Returns a step-by-step explanation of the prediction for an unknown sample.
    """
    explanation = []
    explanation.append(f"Input Features (x): {x}\n")
    
    scores = model.decision_function(x.reshape(1, -1))[0]
    posteriors = model.predict_proba(x.reshape(1, -1))[0]
    
    for i, c in enumerate(model.classes_):
        class_name = target_names[c]
        mean = model.means_[c]
        prior = model.priors_[c]
        
        # Calculate likelihood specifically for explanation
        log_lik = model._log_gaussian_density(x.reshape(1, -1), model.means_[c], model.covariances_[c])[0]
        lik = np.exp(log_lik)
        
        explanation.append(f"Step for Class {class_name}:")
        explanation.append(f"  Prior P(C): {prior:.4f}")
        explanation.append(f"  Log Likelihood log(P(x|C)): {log_lik:.4f}")
        explanation.append(f"  Likelihood P(x|C): {lik:.4e}")
        explanation.append(f"  Joint P(x|C)P(C): {lik * prior:.4e}\n")
        
    explanation.append("Final Normalization (Bayes' Theorem):")
    for i, c in enumerate(model.classes_):
        explanation.append(f"  Posterior P({target_names[c]}|x) = {posteriors[i]:.4f}")
        
    explanation.append(f"\nDiscriminant Scores:")
    for i, c in enumerate(model.classes_):
        explanation.append(f"  g_{target_names[c]}(x) = {scores[i]:.4f}")
        
    pred_idx = np.argmax(posteriors)
    explanation.append(f"\nPredicted Class: {target_names[pred_idx]}")
    
    return "\n".join(explanation)
