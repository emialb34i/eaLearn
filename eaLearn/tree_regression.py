import numpy as np

class Node:
    def __init__(self, feature_i=None, threshold=None, value=None, left=None, right=None):
        self.feature_i = feature_i # feature used for the split
        self.threshold = threshold # threshold for split
        self.value = value # predicted value if node is leaf
        self.left = left # left subtree (true)
        self.right = right # right subtree (false)

class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=float("inf"), min_impurity=1e-7):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_impurity = min_impurity

    def fit(self, X, y):
        # build tree recursively
        self.root = self.build_tree(X,y)
        return self

    def build_tree(self, X, y, current_depth=0):
        best_criteria = None
        best_sets = None
        n_samples, n_features = X.shape
        largest_impurity = np.inf
        # trick to make the dimesions work
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)
        Xy = np.concatenate((X, y), axis=1)

        for feature_i in range(n_features):
            feature_values = X[:,feature_i]
            unique_values = np.unique(feature_values)

            for threshold in unique_values:
                # split on feature and threshold
                Xy1, Xy2 = self.split(Xy, feature_i, threshold)
                if len(Xy1) > 0 and len(Xy2) > 0:
                    y1, y2 = Xy1[:, n_features:], Xy2[:,n_features:] # selct y vals from both sets
                    # for regression we use variance reduction as a measure of impurity
                    impurity = self.variance_reduction(y, y1, y2) 
                    if impurity > largest_impurity:
                        largest_impurity = impurity
                        best_criteria = {'feature_i': feature_i, 'threshold': threshold}
                        best_sets = {
                            'leftX': Xy1[:, :n_features],
                            'lefty': Xy1[:, n_features:],
                            'rightX': Xy2[:, :n_features],
                            'righty': Xy2[:, n_features:] 
                        }
        # check if we should keep building our tree
        if largest_impurity > self.min_impurity or current_depth < self.max_depth or n_samples < self.min_samples_split:
            # build left and right branch (RECURSE!)
            left_branch = self.build_tree(best_sets['leftX'], best_sets['lefty'], current_depth+1)
            right_branch = self.build_tree(best_sets['rightX'], best_sets['righty'], current_depth+1)
            return Node(feature_i=best_criteria['feature_i'], threshold=best_criteria['threshold'], left=left_branch, right=right_branch)
        
        leaf_value = self.mean_value(y)
        
        return Node(value=leaf_value)

    def variance_reduction(self, y, y1, y2):
        # calculate variance reduction
        y_var = y.var()
        y1_var, y2_var = y1.var(), y2.var()
        w1, w2 = len(y1)/len(y), len(y2)/len(y)
        var_reduction = y_var - (w1*y1_var + w2*y2_var)
        return var_reduction

    def predict(self, X):
        # recurse through the tree unitl reaching a leaf and return its value
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

    def precict_value(self, X, tree=None):
        # recursive search down the tree
        if tree is None:
            tree = self.root
        if tree.value is not None:
            return tree.value
        feature_value = X[tree.feature_i]
        subtree = tree.right
        if isinstance(tree.threshold, int) or isinstance(tree.threshold, float):
            if feature_value <= tree.threshold:
                subtree = tree.left
        elif feature_value == tree.threshold:
            subtree = tree.left
        return self.precict_value(X, subtree)

    def mean_value(self, y):
        return np.mean(y)

    def print_tree(self, tree=None, indent=" "):
        """ Recursively print the decision tree """
        if not tree:
            tree = self.root
        # If we're at leaf => print the label
        if tree.value is not None:
            print (tree.value)
        # Go deeper down the tree
        else:
            # Print test
            print ("%s:%s? " % (tree.feature_i, tree.threshold))
            # Print the true scenario
            print ("%sT->" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            # Print the false scenario
            print ("%sF->" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
