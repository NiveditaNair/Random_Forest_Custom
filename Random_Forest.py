import numpy as np

# Load dataset
data = np.genfromtxt('gene_therapy_data.csv', delimiter=',')

# Split data into features and target
X = data[:, :-1] # all columns except the last one
y = data[:, -1] # last column, which contains the target labels

# Define number of trees in the forest
n_trees = 10

# Define maximum depth of each tree
max_depth = 5

# Define minimum number of samples required to split a node
min_samples_split = 2

# Define minimum number of samples required at each leaf node
min_samples_leaf = 1

# Define random seed for reproducibility
random_seed = 42

# Define function to create a random forest
def create_random_forest(X, y, n_trees, max_depth, min_samples_split, min_samples_leaf, random_seed):
    forest = []
    for i in range(n_trees):
        # Select random subset of features for each tree
        n_features = int(np.sqrt(X.shape[1]))
        features = np.random.choice(X.shape[1], n_features, replace=False)
        X_subset = X[:, features]
        # Select random subset of samples for each tree
        sample_indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
        X_sampled = X_subset[sample_indices]
        y_sampled = y[sample_indices]
        # Create decision tree using the sampled data
        tree = create_decision_tree(X_sampled, y_sampled, max_depth, min_samples_split, min_samples_leaf, random_seed)
        forest.append(tree)
    return forest

# Define function to create a decision tree
def create_decision_tree(X, y, max_depth, min_samples_split, min_samples_leaf, random_seed):
    # Define class Node to represent each node in the tree
    class Node:
        def __init__(self, X, y):
            self.X = X
            self.y = y
            self.left = None
            self.right = None
            self.split_feature = None
            self.split_threshold = None
            self.label = None
    # Define function to calculate Gini impurity of a set of labels
    def gini_impurity(labels):
        p0 = np.sum(labels == 0) / len(labels)
        p1 = np.sum(labels == 1) / len(labels)
        return 1 - (p0**2 + p1**2)
    # Define function to split a node into left and right children
    def split_node(node):
        # Calculate Gini impurity of current node
        impurity = gini_impurity(node.y)
        best_impurity = impurity
        best_feature = None
        best_threshold = None
        # Iterate over all features to find best split
        for i in range(node.X.shape[1]):
            # Iterate over all possible thresholds for each feature
            thresholds = np.unique(node.X[:, i])
            for j in range(1, len(thresholds)):
                threshold = (thresholds[j-1] + thresholds[j]) / 2
                left_indices = node.X[:, i] < threshold
                right_indices = node.X[:, i] >= threshold
                left_labels = node.y[left_indices]
                right_labels = node.y[right_indices]
                if len(left_labels) >= min_samples_leaf and len(right_labels) >= min_samples_leaf:
                    left_impurity = gini_impurity(left_labels)
                    right_impurity = gini_impurity(right_labels)
