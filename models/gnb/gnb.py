import numpy as np

class GaussianNaiveBayes:
    """Gaussian Naive Bayes implementation using NumPy"""

    def __init__(self):
        self.means = {}
        self.vars = {}
        self.priors = {}

    def fit(self, X, y):
        """Fit the model by calculating the mean, variance, and prior for each class"""

        classes = np.unique(y)
        n_features = X.shape[1]

        # Initialize mean and variance arrays with zeros (size is num_classes x num_features)
        self.means = np.zeros((len(classes), n_features))
        self.vars = np.zeros((len(classes), n_features))
        # Initialize prior probabilities array with zeros (size is num_classes)
        self.priors = np.zeros(len(classes))

        for class_index, class_label in enumerate(classes):
            
            # Get samples belonging to the current class
            rows_for_class = []
            for i in range(len(y)):
                if y[i] == class_label:
                    rows_for_class.append(X[i])
            X_class = np.array(rows_for_class)

            # Calculate mean and variance for each feature (1e-9 added to avoid division by zero)
            self.means[class_index] = np.mean(X_class, axis=0)
            self.vars[class_index] = np.var(X_class, axis=0) + 1e-9 

            # Calculate prior probability for each class (P(Class) = count(Class) / total_samples)
            self.priors[class_index] = X_class.shape[0] / X.shape[0]

    def _gaussian_probability(self, class_id, x):
        """
        Calculates P(x_i|Class) for a single sample using the Gaussian probability function
        """
        mean = self.means[class_id]
        var = self.vars[class_id]
        
        # Calculate Gaussian probability (P(x_i | Class)
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def predict(self, X):
        """
        Predict the class labels for samples using Bayes rule and then pick the class with the highest posterior probability P(Class|x).
        log(P(Class|x)) = log(P(Class)) + sum log(P(x_i|Class))
        """
        
        y_pred = []

        for x in X:
            posteriors = []

            # For each class, calculate the posterior probability: log(P(Class)) + sum log(P(x_i|Class))
            for i in range(len(self.priors)):
                # Calculate log of prior probability of the class: log(P(Class))
                prior = np.log(self.priors[i])
                # For each feature i in x, calculate Gaussian probability: sum log(P(x_i | Class))
                conditional = np.sum(np.log(self._gaussian_probability(i, x)))
                posterior = prior + conditional
                posteriors.append(posterior)

            # Choose the class with the highest posterior probability
            y_pred.append(np.argmax(posteriors))

        return np.array(y_pred)
    
# Sources:
# https://www.ibm.com/think/topics/naive-bayes