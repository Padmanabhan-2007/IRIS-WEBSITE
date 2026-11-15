import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    
def plot_confusion_matrix(conf_matrix, target_names, output_path=None):
    """
    Plots and saves the confusion matrix heatmap.
    """
    set_style()
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        return plt.gcf()

def plot_feature_distributions(X, y, feature_names, target_names, output_path=None):
    """
    Plots and saves the distributions of all features across classes.
    """
    set_style()
    n_features = X.shape[1]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i in range(n_features):
        for c in np.unique(y):
            sns.kdeplot(X[y == c, i], ax=axes[i], fill=True, label=target_names[c], color=colors[c])
        axes[i].set_title(f'Distribution of {feature_names[i]}')
        axes[i].set_xlabel('Centimeters')
        if i == 0:
            axes[i].legend()
            
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        return fig

def plot_posterior_probabilities(posteriors, target_names, output_path=None):
    """
    Plots the posterior probabilities for a single sample.
    """
    set_style()
    plt.figure(figsize=(8, 5))
    bars = plt.bar(target_names, posteriors, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylim(0, 1.05)
    plt.title('Posterior Probabilities P(C|x)', fontsize=14)
    plt.ylabel('Probability', fontsize=12)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        return plt.gcf()
