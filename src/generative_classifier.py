import numpy as np

class GaussianGenerativeClassifier:
    """
    Probabilistic Generative Classifier using multivariate Gaussian class-conditional densities.
    Demonstrates fundamental concepts of MLE, Bayes theorem, and discriminant functions.
    """
    
    def __init__(self, epsilon=1e-6):
        """
        Args:
            epsilon (float): Regularization term for covariance matrix to ensure numerical stability.
        """
        self.epsilon = epsilon
        self.classes_ = None
        self.means_ = {}
        self.covariances_ = {}
        self.priors_ = {}
        self.d_ = None # Number of features
        
    def fit(self, X, y):
        """
        Fit the generative model using Maximum Likelihood Estimation (MLE).
        
        Args:
            X (np.ndarray): Training features, shape (N, d)
            y (np.ndarray): Training labels, shape (N,)
        """
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)
        N, self.d_ = X.shape
        
        for c in self.classes_:
            # 1. Select samples for this class
            X_c = X[y == c]
            N_c = X_c.shape[0]
            
            # 2. Calculate class prior: P(Ck) = Nk / N
            self.priors_[c] = N_c / N
            
            # 3. Calculate mean vector: mu_k = (1/Nk) * sum(x_i)
            mean_c = np.mean(X_c, axis=0)
            self.means_[c] = mean_c
            
            # 4. Center samples
            X_c_centered = X_c - mean_c
            
            # 5. Calculate covariance matrix using MLE (1/Nk), not unbiased (1/(Nk-1))
            # Sigma_k = (1/Nk) * sum((x_i - mu_k)(x_i - mu_k)^T)
            cov_c = (X_c_centered.T @ X_c_centered) / N_c
            
            # Apply regularization if needed for numerical stability
            cov_c += np.eye(self.d_) * self.epsilon
            
            self.covariances_[c] = cov_c
            
        return self
        
    def _log_gaussian_density(self, X, mean, covariance):
        """
        Calculates the log of the multivariate Gaussian density.
        
        Args:
            X (np.ndarray): Samples, shape (n_samples, d)
            mean (np.ndarray): Mean vector, shape (d,)
            covariance (np.ndarray): Covariance matrix, shape (d, d)
            
        Returns:
            np.ndarray: Log likelihood values, shape (n_samples,)
        """
        n_samples = X.shape[0]
        
        # Calculate determinant using slogdet for numerical stability
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError("Covariance matrix is not positive definite even after regularization.")
            
        # Center the data
        X_centered = X - mean
        
        # Calculate the Mahalanobis distance term: (x-mu)^T * Sigma^-1 * (x-mu)
        # Using linalg.solve instead of explicit inverse: Sigma^-1 * (x-mu) -> solve(Sigma, (x-mu))
        # X_centered is (n_samples, d). We want to solve for each sample.
        # X_centered.T is (d, n_samples). solve(cov, X_centered.T) gives (d, n_samples).
        inv_cov_x = np.linalg.solve(covariance, X_centered.T)
        
        # Now we want (x-mu)^T * inv_cov_x for each sample.
        # X_centered is (n_samples, d) and inv_cov_x is (d, n_samples).
        # We can do row-wise dot product.
        mahalanobis = np.sum(X_centered * inv_cov_x.T, axis=1)
        
        # Log likelihood calculation
        # p(x|C) = 1/((2*pi)^(d/2) * |Sigma|^(1/2)) * exp(-0.5 * mahalanobis)
        # log p(x|C) = -0.5*d*log(2*pi) - 0.5*log(|Sigma|) - 0.5*mahalanobis
        log_density = -0.5 * self.d_ * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * mahalanobis
        
        return log_density

    def predict_log_proba(self, X):
        """
        Predict the log posterior probabilities using Bayes' theorem.
        
        Args:
            X (np.ndarray): Samples, shape (n_samples, d)
            
        Returns:
            np.ndarray: Log posterior probabilities, shape (n_samples, n_classes)
        """
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Initialize array to store log(P(x|C) * P(C))
        log_joint_prob = np.zeros((n_samples, n_classes))
        
        for i, c in enumerate(self.classes_):
            # log(P(x|C))
            log_likelihood = self._log_gaussian_density(X, self.means_[c], self.covariances_[c])
            # log(P(C))
            log_prior = np.log(self.priors_[c])
            # log(P(x|C) * P(C)) = log(P(x|C)) + log(P(C))
            log_joint_prob[:, i] = log_likelihood + log_prior
            
        # To get log posterior: log(P(C|x)) = log_joint_prob - log(sum(exp(log_joint_prob)))
        # Use log-sum-exp trick for numerical stability
        max_log_joint = np.max(log_joint_prob, axis=1, keepdims=True)
        log_marginal = max_log_joint + np.log(np.sum(np.exp(log_joint_prob - max_log_joint), axis=1, keepdims=True))
        
        log_posterior = log_joint_prob - log_marginal
        return log_posterior

    def predict_proba(self, X):
        """
        Predict posterior probabilities P(C|x) for all classes.
        
        Args:
            X (np.ndarray): Samples, shape (n_samples, d)
            
        Returns:
            np.ndarray: Posterior probabilities, shape (n_samples, n_classes)
        """
        log_posterior = self.predict_log_proba(X)
        posterior = np.exp(log_posterior)
        
        # Ensure it sums to exactly 1 over classes due to any minor floating point errors
        posterior = posterior / np.sum(posterior, axis=1, keepdims=True)
        return posterior

    def predict(self, X):
        """
        Predict the class label based on the maximum posterior probability (Bayes decision rule).
        
        Args:
            X (np.ndarray): Samples, shape (n_samples, d)
            
        Returns:
            np.ndarray: Predicted labels, shape (n_samples,)
        """
        posterior = self.predict_proba(X)
        class_indices = np.argmax(posterior, axis=1)
        return self.classes_[class_indices]
        
    def decision_function(self, X):
        """
        Calculates the log discriminant score g_k(x).
        g_k(x) = -0.5 * log(|Sigma_k|) - 0.5 * (x-mu_k)^T Sigma_k^(-1) (x-mu_k) + log(P(C_k))
        This ignores the constant -0.5*d*log(2*pi).
        
        Args:
            X (np.ndarray): Samples, shape (n_samples, d)
            
        Returns:
            np.ndarray: Discriminant scores, shape (n_samples, n_classes)
        """
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        scores = np.zeros((n_samples, n_classes))
        
        for i, c in enumerate(self.classes_):
            # This is exactly log_joint_prob + 0.5*d*log(2*pi)
            log_likelihood = self._log_gaussian_density(X, self.means_[c], self.covariances_[c])
            log_prior = np.log(self.priors_[c])
            scores[:, i] = log_likelihood + log_prior + 0.5 * self.d_ * np.log(2 * np.pi)
            
        return scores
        
    def get_class_statistics(self):
        """
        Returns a dictionary of the learned parameters.
        """
        stats = []
        for c in self.classes_:
            stats.append({
                'class': c,
                'prior': self.priors_[c],
                'mean': self.means_[c],
                'covariance': self.covariances_[c]
            })
        return stats
