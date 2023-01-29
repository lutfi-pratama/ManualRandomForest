import pandas as pd
import numpy as np
import random
import math
import collections
import pickle

from joblib import Parallel, delayed


class Tree(object):
    """Determine Decision Tree"""

    def __init__(self):
        self.split_feature = None
        self.split_value = None
        self.leaf_value = None
        self.tree_left = None
        self.tree_right = None

    def calc_predict_value(self, dataset):
        """Find the leaf node of the sample through the recursive Decision Tree"""
        if self.leaf_value is not None:
            return self.leaf_value
        elif dataset[self.split_feature] <= self.split_value:
            return self.tree_left.calc_predict_value(dataset)
        else:
            return self.tree_right.calc_predict_value(dataset)

    def describe_tree(self):
        """Print the decision tree in the form of json, which is convenient for viewing the tree structure"""
        if not self.tree_left and not self.tree_right:
            leaf_info = "{leaf_value:" + str(self.leaf_value) + "}"
            return leaf_info
        left_info = self.tree_left.describe_tree()
        right_info = self.tree_right.describe_tree()
        tree_structure = "{split_feature:" + str(self.split_feature) + \
            ",split_value:" + str(self.split_value) + \
            ",left_tree:" + left_info + \
            ",right_tree:" + right_info + "}"
        return tree_structure


class RandomForestClassifier(object):
    def __init__(self, n_estimators=10, max_depth=-1, min_samples_split=2, min_samples_leaf=1,
                 min_split_gain=0.0, colsample_bytree=None, subsample=0.8, random_state=None):
        """
        Random Forest Parameters
        ----------
        n_estimators:      Number of trees
        max_depth:         Tree depth, -1 means unlimited depth
        min_samples_split: The minimum number of samples required for node splitting, the node terminates splitting if it is less than this value
        min_samples_leaf:  The minimum number of samples of leaf nodes, less than this value leaves are merged
        min_split_gain:    The minimum gain required for splitting, less than this value the node terminates splitting
        colsample_bytree:  Column sampling setting, which can be [sqrt, log2]. sqrt means randomly selecting sqrt(n_features) features,
                            log2 means to randomly select log(n_features) features, if set to other, column sampling will not be performed
        subsample:         Row Sampling Ratio
        random_state:      Random seed, after setting, the n_estimators sample sets generated each time will not change, ensuring that the experiment can be repeated
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth if max_depth != -1 else float('inf')
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_split_gain = min_split_gain
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.random_state = random_state
        self.trees = None
        self.feature_importances_ = dict()

    def fit(self, dataset, targets):
        """Model Training Entry"""
        assert targets.unique().__len__() == 2, "There must be two class for targets!"
        targets = targets.to_frame(name='label')

        if self.random_state:
            random.seed(self.random_state)
        random_state_stages = random.sample(
            range(self.n_estimators), self.n_estimators)

        # Two Column Sampling Methods
        if self.colsample_bytree == "sqrt":
            self.colsample_bytree = int(len(dataset.columns) ** 0.5)
        elif self.colsample_bytree == "log2":
            self.colsample_bytree = int(math.log(len(dataset.columns)))
        else:
            self.colsample_bytree = len(dataset.columns)

        # Build multiple decision trees in parallel
        self.trees = Parallel(n_jobs=-1, verbose=0, backend="threading")(
            delayed(self._parallel_build_trees)(dataset, targets, random_state)
            for random_state in random_state_stages)

    def _parallel_build_trees(self, dataset, targets, random_state):
        """Bootstrap has put back sampling to generate a training sample set and build a decision tree"""
        subcol_index = random.sample(
            dataset.columns.tolist(), self.colsample_bytree)
        dataset_stage = dataset.sample(n=int(self.subsample * len(dataset)), replace=True,
                                       random_state=random_state).reset_index(drop=True)
        dataset_stage = dataset_stage.loc[:, subcol_index]
        targets_stage = targets.sample(n=int(self.subsample * len(dataset)), replace=True,
                                       random_state=random_state).reset_index(drop=True)

        tree = self._build_single_tree(dataset_stage, targets_stage, depth=0)
        print(tree.describe_tree())
        return tree

    def _build_single_tree(self, dataset, targets, depth):
        """Build decision tree recursively"""
        # If the categories of the node are all the same/the samples are less than the minimum number of samples required for splitting, select the category with the most occurrences. Termination of division
        if len(targets['label'].unique()) <= 1 or dataset.__len__() <= self.min_samples_split:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            return tree

        if depth < self.max_depth:
            best_split_feature, best_split_value, best_split_gain = self.choose_best_feature(
                dataset, targets)
            left_dataset, right_dataset, left_targets, right_targets = \
                self.split_dataset(
                    dataset, targets, best_split_feature, best_split_value)

            tree = Tree()
            # If after the parent node splits, the left leaf node/right leaf node sample is less than the set minimum number of leaf node samples, the parent node will terminate the split
            if left_dataset.__len__() <= self.min_samples_leaf or \
                    right_dataset.__len__() <= self.min_samples_leaf or \
                    best_split_gain <= self.min_split_gain:
                tree.leaf_value = self.calc_leaf_value(targets['label'])
                return tree
            else:
                # If the feature is used when splitting, the importance of the feature is increased by 1
                self.feature_importances_[best_split_feature] = \
                    self.feature_importances_.get(best_split_feature, 0) + 1

                tree.split_feature = best_split_feature
                tree.split_value = best_split_value
                tree.tree_left = self._build_single_tree(
                    left_dataset, left_targets, depth+1)
                tree.tree_right = self._build_single_tree(
                    right_dataset, right_targets, depth+1)
                return tree
        # If the depth of the tree exceeds a preset value, terminate the split
        else:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            return tree

    def choose_best_feature(self, dataset, targets):
        """Find the best data set division method, find the optimal split feature, split threshold, split gain"""
        best_split_gain = 1
        best_split_feature = None
        best_split_value = None

        for feature in dataset.columns:
            if dataset[feature].unique().__len__() <= 100:
                unique_values = sorted(dataset[feature].unique().tolist())
            # If there are too many values for this dimension feature, select the 100th percentile value as the splitting threshold to be selected
            else:
                unique_values = np.unique([np.percentile(dataset[feature], x)
                                           for x in np.linspace(0, 100, 100)])

            # Calculate the splitting gain for the possible splitting thresholds, and select the threshold with the largest gain
            for split_value in unique_values:
                left_targets = targets[dataset[feature] <= split_value]
                right_targets = targets[dataset[feature] > split_value]
                split_gain = self.calc_gini(
                    left_targets['label'], right_targets['label'])

                if split_gain < best_split_gain:
                    best_split_feature = feature
                    best_split_value = split_value
                    best_split_gain = split_gain
        return best_split_feature, best_split_value, best_split_gain

    @staticmethod
    def calc_leaf_value(targets):
        """Select the category with the most occurrences in the sample as the value of the leaf node"""
        label_counts = collections.Counter(targets)
        major_label = max(zip(label_counts.values(), label_counts.keys()))
        return major_label[1]

    @staticmethod
    def calc_gini(left_targets, right_targets):
        """The classification tree uses the Gini index as an indicator to select the optimal split point"""
        split_gain = 0
        for targets in [left_targets, right_targets]:
            gini = 1
            # 统计每个类别有多少样本，然后计算gini
            label_counts = collections.Counter(targets)
            for key in label_counts:
                prob = label_counts[key] * 1.0 / len(targets)
                gini -= prob ** 2
            split_gain += len(targets) * 1.0 / \
                (len(left_targets) + len(right_targets)) * gini
        return split_gain

    @staticmethod
    def split_dataset(dataset, targets, split_feature, split_value):
        """Divide the sample into two parts according to the characteristics and threshold, the left is less than or equal to the threshold, and the right is greater than the threshold"""
        left_dataset = dataset[dataset[split_feature] <= split_value]
        left_targets = targets[dataset[split_feature] <= split_value]
        right_dataset = dataset[dataset[split_feature] > split_value]
        right_targets = targets[dataset[split_feature] > split_value]
        return left_dataset, right_dataset, left_targets, right_targets

    def predict(self, dataset):
        """Input the sample and predict the category it belongs to"""
        res = []
        for _, row in dataset.iterrows():
            pred_list = []
            # Count the prediction results of each tree, and select the result with the most occurrences as the final category
            for tree in self.trees:
                pred_list.append(tree.calc_predict_value(row))

            pred_label_counts = collections.Counter(pred_list)
            pred_label = max(
                zip(pred_label_counts.values(), pred_label_counts.keys()))
            res.append(pred_label[1])
        return np.array(res)


# def load_kuliah_online_pickle():
#     with open(_get_path('source/pickle/XY_train_df.pickle'), 'rb') as infile:
#         XY_train = pickle.load(infile)
#     with open(_get_path('source/pickle/XY_test_df.pickle'), 'rb') as infile:
#         XY_test = pickle.load(infile)

#     return XY_train, XY_test


if __name__ == '__main__':
    df = pd.read_csv("source/wine.txt")
    df = df[df['label'].isin([1, 2])].sample(
        frac=1, random_state=66).reset_index(drop=True)
    clf = RandomForestClassifier(n_estimators=5,
                                 max_depth=5,
                                 min_samples_split=6,
                                 min_samples_leaf=2,
                                 min_split_gain=0.0,
                                 colsample_bytree="sqrt",
                                 subsample=0.8,
                                 random_state=66)

    train_count = int(0.7 * len(df))
    feature_list = ["Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",
                    "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue",
                    "OD280/OD315 of diluted wines", "Proline"]
    clf.fit(df.loc[:train_count, feature_list], df.loc[:train_count, 'label'])

    from sklearn import metrics
    print(metrics.accuracy_score(df.loc[:train_count, 'label'], clf.predict(
        df.loc[:train_count, feature_list])))
    print(metrics.accuracy_score(df.loc[train_count:, 'label'], clf.predict(
        df.loc[train_count:, feature_list])))
