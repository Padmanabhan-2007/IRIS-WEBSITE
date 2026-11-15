import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

# Cell 1
cells.append(nbf.v4.new_markdown_cell("# Iris Probabilistic Generative Classifier\nThis notebook progressive explains the manual implementation of a Gaussian generative classifier."))
cells.append(nbf.v4.new_code_cell("import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.datasets import load_iris\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score, confusion_matrix"))

# Cell 2 & 3
cells.append(nbf.v4.new_markdown_cell("## Load and Inspect Dataset"))
cells.append(nbf.v4.new_code_cell("iris = load_iris()\nX = iris.data\ny = iris.target\nfeature_names = iris.feature_names\ntarget_names = iris.target_names\n\ndf = pd.DataFrame(X, columns=feature_names)\ndf['target'] = [target_names[i] for i in y]\ndf.head()"))

# Cell 4
cells.append(nbf.v4.new_markdown_cell("## Class Distribution"))
cells.append(nbf.v4.new_code_cell("sns.countplot(x='target', data=df)\nplt.title('Class Distribution')\nplt.show()"))

# Cell 5
cells.append(nbf.v4.new_markdown_cell("## Train/Test Split"))
cells.append(nbf.v4.new_code_cell("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)\nprint(f'Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')"))

# Cell 6
cells.append(nbf.v4.new_markdown_cell("## Calculate Class Priors P(C)"))
cells.append(nbf.v4.new_code_cell("classes = np.unique(y_train)\nN = X_train.shape[0]\npriors = {}\n\nfor c in classes:\n    N_c = X_train[y_train == c].shape[0]\n    priors[c] = N_c / N\n    print(f'Prior for {target_names[c]}: {priors[c]:.4f}')"))

# Cell 7
cells.append(nbf.v4.new_markdown_cell("## Calculate Mean Vectors $\mu_k$"))
cells.append(nbf.v4.new_code_cell("means = {}\nfor c in classes:\n    X_c = X_train[y_train == c]\n    means[c] = np.mean(X_c, axis=0)\n    print(f'Mean for {target_names[c]}: {means[c]}')"))

# Cell 8
cells.append(nbf.v4.new_markdown_cell("## Calculate Covariance Matrices $\Sigma_k$\nUsing Maximum Likelihood Estimation ($1/N_k$)"))
cells.append(nbf.v4.new_code_cell("covariances = {}\nfor c in classes:\n    X_c = X_train[y_train == c]\n    N_c = X_c.shape[0]\n    X_c_centered = X_c - means[c]\n    covariances[c] = (X_c_centered.T @ X_c_centered) / N_c\n    print(f'Covariance for {target_names[c]}:\\n{covariances[c]}\\n')"))

# Cell 9 & 10
cells.append(nbf.v4.new_markdown_cell("## Gaussian Density and Likelihoods P(x|C)"))
cells.append(nbf.v4.new_code_cell("def log_gaussian_density(x, mean, cov):\n    d = len(mean)\n    sign, logdet = np.linalg.slogdet(cov)\n    x_centered = x - mean\n    inv_cov_x = np.linalg.solve(cov, x_centered.T)\n    mahalanobis = np.sum(x_centered * inv_cov_x.T, axis=1)\n    return -0.5 * d * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * mahalanobis\n\n# Example flower\nx_example = np.array([[6.0, 2.9, 4.5, 1.5]])\nfor c in classes:\n    log_lik = log_gaussian_density(x_example, means[c], covariances[c])[0]\n    print(f'Likelihood for {target_names[c]}: {np.exp(log_lik):.4e}')"))

# Cell 11 & 12
cells.append(nbf.v4.new_markdown_cell("## Apply Bayes Theorem for Posterior Probabilities P(C|x)"))
cells.append(nbf.v4.new_code_cell("joint_probs = []\nfor c in classes:\n    log_lik = log_gaussian_density(x_example, means[c], covariances[c])[0]\n    joint = np.exp(log_lik) * priors[c]\n    joint_probs.append(joint)\n\nposteriors = joint_probs / np.sum(joint_probs)\nfor c in classes:\n    print(f'Posterior for {target_names[c]}: {posteriors[c]:.4f}')"))

# Cell 13
cells.append(nbf.v4.new_markdown_cell("## Predict Unknown Flower"))
cells.append(nbf.v4.new_code_cell("pred_idx = np.argmax(posteriors)\nprint(f'Predicted class: {target_names[pred_idx]}')"))

# Cell 14
cells.append(nbf.v4.new_markdown_cell("## Evaluate Classifier (using our src package)"))
cells.append(nbf.v4.new_code_cell("import sys\nsys.path.append('..')\nfrom src.generative_classifier import GaussianGenerativeClassifier\n\nmodel = GaussianGenerativeClassifier()\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')"))

# Cell 15
cells.append(nbf.v4.new_markdown_cell("## Show Confusion Matrix"))
cells.append(nbf.v4.new_code_cell("cm = confusion_matrix(y_test, y_pred)\nsns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')\nplt.title('Confusion Matrix')\nplt.ylabel('True Class')\nplt.xlabel('Predicted Class')\nplt.show()"))

# Cell 16
cells.append(nbf.v4.new_markdown_cell("## Conclusion\nWe successfully implemented a Gaussian generative classifier from scratch."))

nb['cells'] = cells
with open(os.path.join(os.path.dirname(__file__), 'notebooks', 'iris_analysis.ipynb'), 'w') as f:
    nbf.write(nb, f)
