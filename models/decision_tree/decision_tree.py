import numpy as np

class DecisionTree:
    def __init__(self, max_depth=50, min_samples_split=2):
        # Maximum depth of the tree
        self.max_depth = max_depth

        # Root of the decision tree
        self.tree = None

        # Minimum samples required to split a node
        self.min_samples_split = min_samples_split

    def _gini_impurity(self, labels):
        """
        Calculate Gini Impurity for a list of class labels 
        Gini Impurity = 1 - Σ (P(class_i))^2
        """
        num_labels = len(labels)
        counts = np.unique(labels, return_counts=True)[1]
        prob = counts / num_labels
        gini = 1 - np.sum(prob ** 2)
        return gini
    
    def _split_data(self, X, y, feature_index, threshold):
        """Split the dataset into left and right subsets based on a feature and threshold"""    
        
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold

        X_left = X[left_mask]
        y_left = y[left_mask]
        X_right = X[right_mask]
        y_right = y[right_mask]

        return X_left, y_left, X_right, y_right

    def _best_split(self, X, y):
        """Find which feature and threshold gives the best split"""
               
        best_gini = 1
        best_feature = None
        best_threshold = None

        for feature_index in range(len(X[0])):
            feature_column = X[:, feature_index]
            values = np.unique(feature_column)

            thresholds = (values[:-1] + values[1:]) / 2 
            for threshold in thresholds:

                X_left, y_left, X_right, y_right = self._split_data(X, y, feature_index, threshold)

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gini_left = self._gini_impurity(y_left)
                gini_right = self._gini_impurity(y_right)
                weighted_gini = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_index
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def fit(self, X, y):
        """Public method to train the decision tree classifier"""
       
        X = np.array(X)
        y = np.array(y)

        self.tree = self._fit(X, y)
    
    def _fit(self, X, y, depth=0):
        """Recursive method to build the decision tree"""
        
        # If all labels are the same, make this a leaf node
        if len(np.unique(y)) == 1:
            return {'value': y[0]}
        
        # If maximum depth is reached or not enough samples to split, make this a leaf node
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            labels, counts = np.unique(y, return_counts=True)
            majority_class = labels[np.argmax(counts)]
            return {'value': majority_class}
        
        feature, threshold = self._best_split(X, y)

        # If no split is found, make this a leaf node
        if feature is None:
            labels, counts = np.unique(y, return_counts=True)
            majority_class = labels[np.argmax(counts)]
            return {'value': majority_class}
        
        X_left, y_left, X_right, y_right = self._split_data(X, y, feature, threshold)
        
        # Repeat recursively for left and right subtrees
        left_subtree = self._fit(X_left, y_left, depth + 1)
        right_subtree = self._fit(X_right, y_right, depth + 1)

        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def _predict_sample(self, sample, node):
        """Predict class of a single sample by growing down the tree"""

        if 'value' in node:
            return node['value']
        
        if sample[node['feature']] <= node['threshold']:
            return self._predict_sample(sample, node['left'])
        else:
            return self._predict_sample(sample, node['right'])

    def predict(self, X):
        """Predict classes for multiple samples"""

        X = np.array(X)
        predictions = []

        for sample in X:
            prediction = self._predict_sample(sample, self.tree)
            predictions.append(prediction)
        
        return np.array(predictions)