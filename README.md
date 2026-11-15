# Iris Classification Using a Probabilistic Generative Model 🌸

## Overview
This is a complete academic Machine Learning project that classifies Iris flowers by modeling each class as a multivariate Gaussian distribution and applying Bayes' theorem to compute posterior probabilities.

The core probabilistic generative classifier is implemented **manually from scratch** using NumPy, bypassing standard `scikit-learn` implementations (like `GaussianNB`) to explicitly demonstrate the mathematical pipeline of Maximum Likelihood Estimation, Gaussian Probability Density Functions, and Bayes' Rule.

## Problem Statement
Build an explainable, probabilistic generative classifier capable of taking four features (sepal length, sepal width, petal length, petal width) and outputting the probability that the flower belongs to one of three classes (Setosa, Versicolor, Virginica).

## Features
- **Generative Modeling:** Models the data distribution directly, not just the decision boundary.
- **Manual Mathematics:** Explicitly calculates the mean vector, covariance matrix, Gaussian likelihood, and Bayesian posterior in raw Python/NumPy.
- **Numerical Stability:** Uses `np.linalg.solve` and `np.linalg.slogdet` and log-space operations to prevent floating-point underflow.
- **Interactive UI:** Professional Streamlit dashboard for predicting new flowers and visualizing class statistics.
- **Explainability:** Step-by-step mathematical tracing of predictions (useful for vivas/presentations).

## Technology Stack
- **Language:** Python 3.11+
- **Core Math:** NumPy
- **Data Manipulation:** Pandas
- **Metrics/Data Loading:** Scikit-learn
- **Visualizations:** Matplotlib, Seaborn
- **UI:** Streamlit
- **Testing:** Pytest

## Dataset
The classic Fisher's Iris Dataset.
- **Features:** Sepal Length, Sepal Width, Petal Length, Petal Width (cm).
- **Classes:** Setosa, Versicolor, Virginica.
- **Size:** 150 samples (50 per class).

## Mathematical Model
1. **Prior:** $P(C_k) = N_k / N$
2. **Mean Vector:** $\mu_k = \frac{1}{N_k} \sum_{i=1}^{N_k} x_i$
3. **Covariance Matrix:** $\Sigma_k = \frac{1}{N_k} \sum_{i=1}^{N_k} (x_i - \mu_k)(x_i - \mu_k)^T$
4. **Gaussian Likelihood:** $P(x | C_k) = \frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}} \exp\left( -\frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) \right)$
5. **Posterior:** $P(C_k | x) = \frac{P(x | C_k) P(C_k)}{\sum_{j=1}^K P(x | C_j) P(C_j)}$

## Architecture & Project Structure
```
iris-probabilistic-generative-classifier/
├── app/                  # Streamlit UI
├── data/                 # Raw/processed data
├── notebooks/            # Jupyter notebooks for interactive analysis
├── outputs/              # Generated plots and statistics
├── report/               # Academic reports and viva questions
├── src/                  # Core source code (model, evaluation, visualization)
├── tests/                # Pytest unit tests
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## Installation
1. Clone this repository or open the project folder.
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Project

### 1. Run the CLI Application
The CLI application trains the model, evaluates it, and generates output charts to the `outputs/` folder.
```bash
python src/main.py
```

### 2. Run the Streamlit Application
Launch the interactive dashboard to visualize the math and predict custom flower measurements.
```bash
streamlit run app/streamlit_app.py
```

### 3. Run Tests
Verify the mathematical constraints (e.g., probabilities summing to 1, matrix dimensions).
```bash
pytest
```

## Example Prediction
Input: `x = [6.0, 2.9, 4.5, 1.5]`
1. Model computes likelihood under Setosa, Versicolor, and Virginica.
2. Applies prior probabilities.
3. Normalizes to get the exact posterior probabilities.
4. Outputs the predicted class (typically Versicolor).

## Academic Mappings

### Unit 1 Mapping
- **Machine Learning**: Complete classification pipeline.
- **Supervised Learning**: Training on the labeled Iris dataset.
- **Continuous Random Variables**: The 4 flower measurements.
- **Mean & Variance**: Handled via $\mu$ and diagonal of $\Sigma$.
- **Bayes Rule**: Applied to compute posterior $P(C|x)$.

### Unit 2 Mapping
- **Maximum Likelihood Estimation (MLE)**: Used to derive parameters. Notice the use of $1/N$ instead of $1/(N-1)$ for the covariance estimator.
- **Probabilistic Generative Model**: The class-conditional Gaussian model $p(x|C)$.
- **Discriminant Function**: Log-space discriminant computation $g_k(x)$ for numerical stability.

## Future Improvements
- Add support for shared covariance (Linear Discriminant Analysis).
- Add functionality to upload custom CSV datasets.
- Support for regularized covariance structures (e.g., shrinkage).

## Web UI (FastAPI + canvas)

A custom front end replaces the Streamlit demo. The browser talks to the *same* hand-written
`GaussianGenerativeClassifier` through a small FastAPI bridge.

```bash
pip install -r requirements.txt
python -m uvicorn backend.server:app --port 8000   # or double-click run_web.bat
# open http://localhost:8000
```

| Endpoint | Purpose |
|---|---|
| `POST /api/predict` | priors, log-likelihoods, Mahalanobis d², discriminants, posteriors for one flower |
| `GET /api/meta` | feature ranges, per-class prior / mean / covariance, fit time |
| `GET /api/field?fx=&fy=` | exact 2-D marginal posterior map for the decision-field plate |
| `GET /api/evaluation` | test accuracy, confusion matrix, per-class P/R/F1, every test specimen |
| `GET /api/dataset` | all 150 samples with true/predicted class |

The 3D iris is a WebGL scene (`web/js/stage.js`, Three.js): smooth parametric petals with a physical
material, soft shadows and image-based light. Falls follow the sepal measurements, standards follow the
petal measurements, and the page's background colour follows the posterior.
