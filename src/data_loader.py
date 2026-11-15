import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os

def load_and_prepare_data(test_size=0.3, random_state=42, save_csv=True, output_dir='data'):
    """
    Loads the Iris dataset, prepares it, and optionally saves to CSV.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names, target_names
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Optional: Save to CSV for the academic report
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        df['class_name'] = df['target'].map({i: name for i, name in enumerate(target_names)})
        df.to_csv(os.path.join(output_dir, 'iris.csv'), index=False)
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_names, target_names

def get_full_dataset():
    """
    Returns the full dataset as X, y, feature_names, target_names
    """
    iris = load_iris()
    return iris.data, iris.target, iris.feature_names, iris.target_names
