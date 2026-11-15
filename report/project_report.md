# Iris Classification Using a Probabilistic Generative Model

## 1. Title
Iris Classification Using a Probabilistic Generative Model

## 2. Abstract
This project implements a custom probabilistic generative classifier from scratch in Python to classify the Iris dataset. By modeling each class using a multivariate Gaussian distribution, the project explicitly demonstrates core statistical and probability concepts including Maximum Likelihood Estimation, class-conditional densities, Bayes' theorem, and posterior probabilities.

## 3. Problem Statement
The goal is to accurately classify Iris flowers into three species based on four physical measurements. Rather than utilizing pre-packaged classification models, the problem demands building the mathematical pipeline manually to explicitly demonstrate how generative models evaluate class likelihoods and compute Bayesian posterior probabilities.

## 4. Motivation
Understanding the mathematical foundations of machine learning is critical. Generative models offer explainability by not just drawing decision boundaries, but by modeling the data distribution itself. This project aims to bridge academic probability theory with practical machine learning implementation.

## 5. Objectives
- Implement a Multivariate Gaussian Generative Classifier.
- Manually calculate mean vectors and covariance matrices using Maximum Likelihood Estimation.
- Apply Bayes' theorem to compute posterior probabilities.
- Provide a clear, step-by-step mathematical explanation of the prediction pipeline.

## 6. Dataset Description
The Iris dataset contains 150 samples across 3 classes (Setosa, Versicolor, Virginica), each with 50 samples.
**Features (Continuous Random Variables):**
1. Sepal Length (cm)
2. Sepal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)

## 7. Unit 1 Concepts Used
| Concept | Project Usage |
|---------|---------------|
| Machine Learning | The overall classification system |
| Supervised Learning | Utilizing the labeled Iris dataset for training |
| Multiclass Classification | Differentiating between three flower classes |
| Continuous Random Variables | The four physical flower measurements |
| Probability Density Function | Modeling features using Gaussian density |
| Mean / Expectation | Calculating the class mean vector |
| Variance | Diagonal entries of the covariance matrix |
| Covariance | Relationships between features |
| Bayes Rule | Computing the posterior probability |

## 8. Unit 2 Concepts Used
| Concept | Project Usage |
|---------|---------------|
| Maximum Likelihood Estimation | Estimating $\mu$ and $\Sigma$ parameters |
| Discriminant Function | Calculating class log discriminant scores |
| Probabilistic Generative Model| Modeling the data distribution $p(x|C)$ |
| Class Prior | Calculating $P(C)$ |
| Likelihood | Calculating $P(x|C)$ using the Gaussian PDF |
| Bayesian Posterior | Computing $P(C|x)$ for final decisions |
| Decision Rule | Selecting the class via argmax posterior |

## 9. Proposed Methodology
The project proposes using a class-conditional Gaussian model. During training, the data is split by class to calculate the prior probability, mean vector, and covariance matrix. During prediction, the likelihood of a new sample under each class distribution is calculated and multiplied by the prior to yield the posterior probability.

## 10. System Architecture
1. **Data Loading:** Train/Test split and stratification.
2. **Parameter Estimation:** MLE computation for priors, means, and covariances.
3. **Inference Engine:** Likelihood calculation -> Bayes theorem -> Posterior normalization.
4. **Evaluation Module:** Accuracy, Confusion Matrix.
5. **Presentation:** Streamlit interactive UI.

## 11. Mathematical Formulation
Let the features be a vector $x \in \mathbb{R}^4$. Let the classes be $C_k \in \{C_1, C_2, C_3\}$.

## 12. Maximum Likelihood Estimation
We estimate parameters assuming the data within each class is drawn from a Gaussian distribution. We maximize the log-likelihood of the observed data.

## 13. Mean Estimation
$$\mu_k = \frac{1}{N_k} \sum_{i=1}^{N_k} x_i$$

## 14. Covariance Estimation
$$\Sigma_k = \frac{1}{N_k} \sum_{i=1}^{N_k} (x_i - \mu_k)(x_i - \mu_k)^T$$
*(Note: We use $1/N_k$ as the MLE, not the unbiased estimator $1/(N_k-1)$)*

## 15. Gaussian Probability Density
The likelihood is defined by the multivariate Gaussian probability density function (PDF):
$$P(x | C_k) = \frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}} \exp\left( -\frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) \right)$$

## 16. Bayes Theorem
$$P(C_k | x) = \frac{P(x | C_k) P(C_k)}{P(x)}$$

## 17. Posterior Probability
$$P(C_k | x) = \frac{P(x | C_k) P(C_k)}{\sum_{j=1}^K P(x | C_j) P(C_j)}$$

## 18. Discriminant Function
To avoid underflow, we work in the log space. The log discriminant score simplifies to:
$$g_k(x) = -\frac{1}{2} \ln |\Sigma_k| - \frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \ln P(C_k)$$

## 19. Algorithm
1. Split dataset into train/test sets.
2. For each class $k$, compute $N_k$, $P(C_k)$, $\mu_k$, and $\Sigma_k$.
3. For a test point $x$, compute $g_k(x)$ for all $k$.
4. Predict $\hat{C} = \arg\max_k g_k(x)$.

## 20. Implementation
The model is implemented in Python utilizing NumPy for matrix operations. Operations like matrix inversion are replaced by more numerically stable equivalents like `np.linalg.solve` and `np.linalg.slogdet`. A small regularization term $\epsilon I$ is added to the covariance matrix to ensure it remains positive definite.

## 21. Experimental Setup
- Language: Python 3.11+
- Test Split: 30%
- Core Libraries: NumPy, Pandas, Scikit-learn (metrics only).
- UI Framework: Streamlit.

## 22. Results
The model effectively classifies the Iris dataset, achieving excellent accuracy comparable to standard library implementations, validating the mathematical formulas.

## 23. Confusion Matrix
The confusion matrix (saved in `outputs/confusion_matrix.png`) demonstrates that Setosa is linearly separable, while a slight overlap exists between Versicolor and Virginica, reflecting the natural feature distributions.

## 24. Unknown Flower Prediction
Given $x = [6.0, 2.9, 4.5, 1.5]$, the model calculates the likelihoods under all three classes, applies the priors, normalizes them, and consistently predicts **Versicolor**, displaying the exact posterior probabilities step-by-step.

## 25. Advantages
- Deeply explainable; every parameter has physical meaning.
- Naturally handles multi-class problems.
- Models the data distribution directly.

## 26. Limitations
- Assumes features are normally distributed.
- The quadratic boundary can be overly sensitive if covariance matrices are ill-conditioned.

## 27. Future Scope
- Implement shared covariance (Linear Discriminant Analysis) vs per-class covariance (Quadratic Discriminant Analysis).
- Extend the application to handle missing data via the generative framework.

## 28. Conclusion
This project successfully bridged theoretical Machine Learning concepts from Unit 1 and Unit 2 with practical implementation. By manually calculating expectations, variances, Gaussian likelihoods, and Bayesian posteriors, the true mechanics of a probabilistic generative model were explored.

## 29. References
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Duda, R. O., Hart, P. E., & Stork, D. G. (2000). *Pattern Classification*. Wiley.
