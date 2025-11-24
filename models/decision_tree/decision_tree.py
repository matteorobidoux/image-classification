import numpy as np

class DecisionTree:
    """
    Decision Tree model that uses Gini Impurity as the splitting criterion.
    """
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.tree = None

    def _gini_impurity(self, labels):
        """
        Calculate Gini Impurity for a list of class labels
        Gini Impurity = 1 - sum (P(class_i))^2
        """
        num_labels = len(labels)

        # Amount of occurrences of each class
        class_counts = np.unique(labels, return_counts=True)[1]
        
        # Probability of each class
        class_probabilities  = class_counts / num_labels
        
        # Gini Impurity calculation: (1 - sum (P(class_i))^2)
        gini = 1 - np.sum(class_probabilities  ** 2)
        
        return gini
    
    def _split_data(self, X, y, feature_index, threshold):
        """Split the dataset into left and right subsets based the threshold"""    
        
        X_left, y_left, X_right, y_right = [], [], [], []

        # Split the data based on the threshold
        for i in range(len(X)):
            if X[i][feature_index] <= threshold:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])

        X_left = np.array(X_left)
        y_left = np.array(y_left)
        X_right = np.array(X_right)
        y_right = np.array(y_right)


        return X_left, y_left, X_right, y_right

    def _best_split(self, X, y):
        """Find which feature and threshold gives the best split (lowest Gini Impurity)"""
               
        best_gini = 1
        best_feature_index = None
        best_threshold = None

        # Iterate through all features
        for feature_index in range(len(X[0])):
            feature_column = X[:, feature_index]
            feature_values = np.unique(feature_column)

            # Try every feature and every unique value as a possible split, then pick the split with the lowest weighted Gini impurity
            for value in feature_values:

                # Split the data
                X_left, y_left, X_right, y_right = self._split_data(X, y, feature_index, value)

                # Skip if one side is empty
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gini_left = self._gini_impurity(y_left)
                gini_right = self._gini_impurity(y_right)

                # Weighted Gini Impurity: (n_left * Gini_left + n_right * Gini_right) / n_total
                weighted_gini = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)

                # Update best split if current is better
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature_index = feature_index
                    best_threshold = value
        
        return best_feature_index, best_threshold
    
    def fit(self, X, y):
        """Used to train the decision tree model"""
       
        X = np.array(X)
        y = np.array(y)

        self.tree = self._build_tree(X, y, 0)
    
    def _build_tree(self, X, y, depth):
        """Build decision tree recursively"""
        
        # If max depth is reached or theres only 1 class, make this a leaf node
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            labels, counts = np.unique(y, return_counts=True)
            majority_class = labels[np.argmax(counts)]
            return {'value': majority_class}
        
        # Find the best feature and threshold to split on
        feature_index, threshold = self._best_split(X, y)

        # If no split is found, make this a leaf node
        if feature_index is None:
            labels, counts = np.unique(y, return_counts=True)
            majority_class = labels[np.argmax(counts)]
            return {'value': majority_class}
        
        # Split the data
        X_left, y_left, X_right, y_right = self._split_data(X, y, feature_index, threshold)
        
        # Repeat recursively for left and right subtrees
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        # Return decision node
        return {
            'feature_index': feature_index,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def _predict_sample(self, sample, node):
        """Predict class of a sample by going through the tree"""

        # If leaf node, return the class value
        if 'value' in node:
            return node['value']
        
        # Go left or right based on the samples feature value to the threshold
        if sample[node['feature_index']] <= node['threshold']:
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
    
# Sources:
# https://blog.quantinsti.com/gini-index/
# https://koalaverse.github.io/machine-learning-in-R/decision-trees.html#tree-algorithms
# https://medium.com/@enozeren/building-a-decision-tree-from-scratch-324b9a5ed836