import numpy as np

class GaussianNaiveBayes:
    """Simple Gaussian Naive Bayes implementation using NumPy."""

    def __init__(self):
        self.means = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        """Fit the model by calculating the mean, variance, and prior for each class"""

        # Get all unique class labels (Cause values)
        classes = np.unique(y)
        n_features = X.shape[1]

        self.means = np.zeros((len(classes), n_features))
        self.vars = np.zeros((len(classes), n_features))
        self.priors = np.zeros(len(classes))

        for idx, cls in enumerate(classes):
            X_c = X[y == cls]

            # Calculate mean and variance for each feature
            self.means[idx, :] = X_c.mean(axis=0)
            self.vars[idx, :] = X_c.var(axis=0) + 1e-9

            # Calculate prior probability for each class (P(Cause) = count(Cause) / total_samples)
            self.priors[idx] = X_c.shape[0] / X.shape[0]

    def _gaussian_probability(self, class_idx, x):
        """
        Calculates P(x|Cause) for a single sample using the Gaussian probability function.
        P(x_i | Class) = (1 / sqrt(2πσ²)) * exp(-(x_i - μ)² / (2σ²))
        """
        
        mean = self.means[class_idx]
        var = self.vars[class_idx]
        
        # Calculate Gaussian probability
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def predict(self, X):
        """
        Predict the class labels for given samples using Bayes rule and then pick the class with the highest posterior probability P(Cause|x).
        P(Cause|x) = P(Cause) * Π P(x_i | Cause)
        Using log probabilities to avoid underflow:
        log(P(Cause|x)) = log(P(Cause)) + Σ log(P(x_i | Cause))
        """
        
        y_pred = []
        for x in X:
            posteriors = []

            # For each class, calculate: log(P(Cause)) + Σ log(P(x_i | Cause))
            for idx in range(len(self.priors)):
                prior = np.log(self.priors[idx])
                conditional = np.sum(np.log(self._gaussian_probability(idx, x)))
                posterior = prior + conditional
                posteriors.append(posterior)

            # Choose the class with the highest posterior probability
            y_pred.append(np.argmax(posteriors))

        return np.array(y_pred)