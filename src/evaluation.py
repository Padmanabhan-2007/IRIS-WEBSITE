from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

def evaluate_model(model, X_test, y_test, target_names=None):
    """
    Evaluates the trained model on test data.
    
    Args:
        model: Trained classifier.
        X_test: Test features.
        y_test: True labels.
        target_names: Optional human-readable class names.
        
    Returns:
        dict: A dictionary containing accuracy, confusion matrix, and classification report.
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    if target_names is not None:
        report = classification_report(y_test, y_pred, target_names=target_names)
    else:
        report = classification_report(y_test, y_pred)
        
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
        'y_pred': y_pred
    }

def generate_class_statistics_df(model, target_names, feature_names):
    """
    Generates a DataFrame of class statistics.
    """
    stats = model.get_class_statistics()
    data = []
    
    for s in stats:
        row = {
            'class': target_names[s['class']] if target_names is not None else s['class'],
            'prior': s['prior']
        }
        for i, f_name in enumerate(feature_names):
            row[f'mean_{f_name.replace(" (cm)", "").replace(" ", "_")}'] = s['mean'][i]
            
        data.append(row)
        
    return pd.DataFrame(data)
