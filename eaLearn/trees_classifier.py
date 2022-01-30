import numpy as np

class Node:
    def __init__(self, feature_i=None, threshold=None, value=None, left=None, right=None):
        self.feature_i = feature_i # feature used for the split
        self.threshold = threshold # threshold for split
        self.value = value # predicted value if node is leaf
        self.left = left # left subtree (true)
        self.right = right # right subtree (false)

class DecisionTreeClassification:
    def __init__(self, min_samples_split=2, max_depth=float("inf"), min_impurity=1e-7):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_impurity = min_impurity
        
    def fit(self, X, y):
        # build tree recursively
        self.root = self.build_tree(X,y)
        return self
    
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

    def predict(self, X):
        # recurse through the tree unitl reaching a leaf and return its value
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

    def build_tree(self, X, y, current_depth=0):
        best_criteria = None
        best_sets = None
        n_samples, n_features = X.shape
        largest_impurity = 0
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
                    # calulculate the gini gain
                    y1, y2 = Xy1[:, n_features:], Xy2[:,n_features:] # selct y vals from both sets
                    impurity = self.gini_gain(y, y1, y2)
                    # compare gini gain of current split
                    # if gini gain the best so far then make the best split
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
        
        leaf_value = self.majority_vote(y)
        
        return Node(value=leaf_value)
    
    def split(self, X, feature, threshold):
        split_func = None
        if isinstance(threshold, int) or isinstance(threshold, float):
            split_func = lambda sample: sample[feature] <= threshold
        else:
            split_func = lambda sample: sample[feature] == threshold
        X1 = np.array([sample for sample in X if split_func(sample)])
        X2 = np.array([sample for sample in X if not split_func(sample)])
        return X1, X2
    
    def gini(self, y):
        m = len(y)
        _, counts = np.unique(y, return_counts=True)
        gini = 1 - np.sum((counts/m)**2)
        return gini

    def gini_gain(self, y, y1, y2):
        """
        y  : y values from parent node
        y1 : y1 values from left child
        y2 : y2 values from right child
        """
        parent_gini = self.gini(y)
        left_gini, right_gini = self.gini(y1), self.gini(y2)
        left_w, right_w = len(y1)/len(y), len(y2)/len(y)
        gini_gain = parent_gini - (left_gini*left_w + right_gini*right_w)
        return gini_gain
    
    def majority_vote(self, y):
        values, counts = np.unique(y, return_counts=True)
        ind = np.argmax(counts)
        return values[ind]
    
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
            
