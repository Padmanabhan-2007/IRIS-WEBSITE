import sys
import os
import numpy as np

# Ensure src module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_and_prepare_data
from src.generative_classifier import GaussianGenerativeClassifier
from src.evaluation import evaluate_model, generate_class_statistics_df
from src.visualizations import plot_confusion_matrix, plot_feature_distributions, plot_posterior_probabilities
from src.utils import explain_prediction

def main():
    print("="*60)
    print(" IRIS CLASSIFICATION USING A PROBABILISTIC GENERATIVE MODEL")
    print("="*60)
    
    # 1. Load dataset
    print("\n[1] Loading dataset...")
    X_train, X_test, y_train, y_test, feature_names, target_names = load_and_prepare_data(
        save_csv=True, output_dir='data'
    )
    print(f"    Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 2. Train classifier
    print("\n[2] Training Generative Classifier (MLE)...")
    model = GaussianGenerativeClassifier()
    model.fit(X_train, y_train)
    
    # Save statistics
    stats_df = generate_class_statistics_df(model, target_names, feature_names)
    os.makedirs('outputs', exist_ok=True)
    stats_df.to_csv('outputs/class_statistics.csv', index=False)
    
    # Print learned parameters
    print("\n--- Learned Parameters ---")
    for i, c in enumerate(model.classes_):
        class_name = target_names[c]
        print(f"\nClass: {class_name.upper()}")
        print(f"  Prior P(C): {model.priors_[c]:.4f}")
        print(f"  Mean Vector: {np.round(model.means_[c], 4)}")
        print("  Covariance Matrix:")
        for row in model.covariances_[c]:
            print("    [" + "  ".join([f"{val:6.4f}" for val in row]) + "]")
            
    # 3. Evaluation
    print("\n[3] Evaluating on Test Set...")
    eval_results = evaluate_model(model, X_test, y_test, target_names)
    print(f"    Accuracy: {eval_results['accuracy'] * 100:.2f}%")
    print("\nClassification Report:")
    print(eval_results['classification_report'])
    
    # Generate visual outputs
    plot_confusion_matrix(eval_results['confusion_matrix'], target_names, 'outputs/confusion_matrix.png')
    plot_feature_distributions(np.vstack((X_train, X_test)), np.concatenate((y_train, y_test)), 
                               feature_names, target_names, 'outputs/feature_distributions.png')
                               
    # 4. Unknown Flower Prediction
    print("\n[4] Example: Predicting Unknown Flower...")
    # Example flower [6.0, 2.9, 4.5, 1.5] - typically Versicolor
    x_unknown = np.array([6.0, 2.9, 4.5, 1.5])
    print(explain_prediction(x_unknown, model, target_names))
    
    # Save posterior probabilities plot
    posteriors = model.predict_proba(x_unknown.reshape(1, -1))[0]
    plot_posterior_probabilities(posteriors, target_names, 'outputs/posterior_probabilities.png')
    
    print("\n" + "="*60)
    print(" Execution Complete. Outputs saved to /outputs/")
    print("="*60)

if __name__ == "__main__":
    main()
