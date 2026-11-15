# Viva Questions: Probabilistic Generative Models

1. **What is supervised learning?**
   Supervised learning is a machine learning paradigm where the algorithm is trained on a labeled dataset, mapping input features to known output targets.

2. **Why is Iris classification supervised learning?**
   Because the dataset provides explicit target labels (the flower species) for every set of input measurements, allowing the model to learn the relationship between features and the target species.

3. **Why is this classification and not regression?**
   Because the target variable (species) is categorical (Setosa, Versicolor, Virginica), whereas regression is used for predicting continuous numerical values.

4. **Why are there three classes?**
   The Iris dataset inherently contains three specific species of Iris flowers: Iris Setosa, Iris Versicolor, and Iris Virginica.

5. **Why use a Gaussian distribution?**
   The Gaussian (Normal) distribution is a mathematically convenient and naturally occurring distribution that models continuous data centered around a mean with specific variance, fitting the continuous measurements of flower petals and sepals well.

6. **What is a probability density function?**
   A PDF describes the relative likelihood for a continuous random variable to take on a given value. The area under the PDF curve equals 1.

7. **What is a prior probability?**
   The prior probability, $P(C)$, is our belief or the initial probability that an observation belongs to a specific class before considering any of its feature measurements.

8. **What is likelihood?**
   Likelihood, $P(x|C)$, is the probability of observing the specific feature values $x$ given that the sample belongs to class $C$.

9. **What is posterior probability?**
   The posterior probability, $P(C|x)$, is the updated probability that the sample belongs to class $C$ after considering its feature measurements $x$.

10. **State Bayes theorem.**
    $P(C|x) = \frac{P(x|C) \cdot P(C)}{P(x)}$

11. **What is a class-conditional density?**
    It is the probability distribution of the input features $x$ for a specific class $C$, represented as $P(x|C)$.

12. **Why calculate a mean vector?**
    The mean vector $\mu$ represents the "center" or expected value of the features for a particular class in the multi-dimensional feature space.

13. **Why calculate a covariance matrix?**
    The covariance matrix $\Sigma$ describes both the variance of each individual feature and how the features vary together (correlate) within a class.

14. **What does diagonal covariance represent?**
    The diagonal entries represent the variance of each individual feature, assuming no correlation between different features.

15. **What does off-diagonal covariance represent?**
    The off-diagonal entries represent the covariance (correlation direction and magnitude) between pairs of different features.

16. **Why is the covariance matrix 4x4?**
    Because the Iris dataset has 4 features (sepal length, sepal width, petal length, petal width). The matrix represents the relationship between all $4 \times 4$ combinations of features.

17. **What is Maximum Likelihood Estimation?**
    MLE is a method of estimating the parameters (like mean and covariance) of a probability distribution by maximizing a likelihood function, making the observed data most probable.

18. **Why use $1/N$ instead of $1/(N-1)$?**
    $1/N$ is the exact mathematical derivation for the Maximum Likelihood Estimate (MLE) of the covariance matrix, whereas $1/(N-1)$ is Bessel's correction to make the estimator unbiased. Since we focus on MLE, we use $1/N$.

19. **What is a discriminant function?**
    A function $g(x)$ that takes an input vector $x$ and assigns it a score for a class. The classifier predicts the class with the highest score.

20. **What is Mahalanobis distance?**
    It is a multi-dimensional measure of distance between a point and a distribution, taking into account the covariance (shape) of the distribution: $\sqrt{(x-\mu)^T \Sigma^{-1} (x-\mu)}$.

21. **Why use the log probability?**
    Multiplying many small probabilities leads to numerical underflow. Logarithms turn multiplication into addition, keeping the numbers numerically stable.

22. **Why use `slogdet`?**
    `numpy.linalg.slogdet` calculates the sign and logarithm of the determinant in a stable way, avoiding overflow or underflow that can occur if you calculate the determinant directly and then take the log.

23. **Why use `solve()`?**
    Computing the explicit inverse of a matrix $\Sigma^{-1}$ is computationally expensive and prone to precision errors. Solving the linear system $\Sigma x = b$ using `numpy.linalg.solve` is much more stable and efficient.

24. **Why not use GaussianNB?**
    `GaussianNB` from scikit-learn assumes that all features are conditionally independent (i.e., a diagonal covariance matrix). Our model uses a full covariance matrix, making it more expressive. Moreover, building it from scratch demonstrates understanding of the math.

25. **What is the difference between generative and discriminative models?**
    Generative models learn the joint probability $P(x, y)$ (or $P(x|C)P(C)$) by modeling the distribution of the data. Discriminative models learn the decision boundary or $P(y|x)$ directly without modeling the underlying distribution.

26. **What happens if covariance is singular?**
    If singular, the matrix cannot be inverted (determinant is 0), and the Gaussian PDF formula fails. This can happen with highly correlated features or insufficient data. We add a small value $\epsilon$ to the diagonal (regularization) to prevent this.

27. **Why do posterior probabilities sum to 1?**
    Because of the denominator in Bayes' theorem (the law of total probability), which normalizes the values so that the sample must belong to one of the defined classes.

28. **What is the final decision rule?**
    Assign the sample to the class $C_k$ that maximizes the posterior probability $P(C_k | x)$ (or equivalently, the log discriminant score $g_k(x)$).

29. **What does confidence mean here?**
    Confidence can be interpreted as the magnitude of the posterior probability $P(C_k|x)$ for the predicted class. A value near 1.0 indicates high certainty under the model assumptions.

30. **What are limitations of the Gaussian assumption?**
    If the real data distribution is heavily skewed, multimodal, or has heavy tails, the Gaussian distribution will be a poor fit, leading to sub-optimal classification boundaries.
