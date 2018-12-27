import numpy as np
import pandas as pd
from math import log
import data_sets_utils as utils


class Node:
    def __init__(self, root=True, label=None, feature_name=None, feature=None):
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.result = {'label:': self.label, 'feature': self.feature, 'tree': self.tree}

    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self, val, node):
        self.tree[val] = node

    def predict(self, features):
        try:
            if self.root is True:
                return self.label
            return self.tree[features[self.feature]].predict(features)
        except:
            return "Other"


class DecisionTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    def fit(self, train_data):
        _, y_train, features = train_data.iloc[:, :-1], train_data.iloc[:, -1], train_data.columns[:-1]
        # print(_)
        # print("-------------")
        # print(y_train)
        # print("-------------")
        # print(train_data.columns)
        # print(train_data.columns[-3:-1])
        # print(features)
        # print("y_train.iloc[0]:")
        # print(y_train.iloc[0])
        self._tree = self.train(train_data)
        return self._tree

    def predict(self, test_data):
        return self._tree.predict(test_data)

    def train(self, train_data):
        _, y_train, features = train_data.iloc[:, :-1], train_data.iloc[:, -1], train_data.columns[:-1]
        # Single Node Tree
        if len(y_train.value_counts()) == 1:
            return Node(root=True, label=y_train.iloc[0])
        # Single Node Tree. Return the Max as the node class Tag.
        if len(features) == 0:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])
        # Calculate the maximum information gain
        max_feature, max_info_gain = self.info_gain_train(np.array(train_data))
        max_feature_name = features[max_feature]

        if max_info_gain < self.epsilon:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        node_tree = Node(root=False, feature_name=max_feature_name, feature=max_feature)

        feature_list = train_data[max_feature_name].value_counts().index
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop([max_feature_name], axis=1)
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)
            # print(f)
            # print(sub_tree)
        return node_tree

    def info_gain_train(self, data_sets):
        count = len(data_sets[0]) - 1
        ent = self.calc_ent(data_sets)
        best_feature = []
        for c in range(count):
            c_info_gain = self.info_gain(ent, self.cond_ent(data_sets, axis=c))
            best_feature.append((c, c_info_gain))
        best = max(best_feature, key=lambda x: x[-1])
        return best

    # Entropy
    @staticmethod
    def calc_ent(data_sets):
        data_length = len(data_sets)
        label_count = {}
        for i in range(data_length):
            label = data_sets[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        ent = -sum([(p / data_length) * log(p / data_length, 2) for p in label_count.values()])
        return ent

    # Information gain
    @staticmethod
    def info_gain(ent, cond_ent):
        return ent - cond_ent

    # Empirical conditional entropy
    def cond_ent(self, data_sets, axis):
        data_length = len(data_sets)
        feature_sets = {}
        for i in range(data_length):
            feature = data_sets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(data_sets[i])
        cond_ent = sum([(len(p) / data_length) * self.calc_ent(p) for p in feature_sets.values()])
        return cond_ent


def predict_data(dt, iris_data_test_df, iris_labels):
    features_data_array = np.array(iris_data_test_df.iloc[:, :-1])
    # print(features_data_array)
    predict_result = []
    for item in features_data_array:
        data = item[:]
        # print(data)
        predict_label = dt.predict(data)
        data = np.append(data, predict_label)
        predict_result.append(data)
    print(predict_result)
    predict_result_df = pd.DataFrame(predict_result, columns=iris_labels)
    return predict_result_df


def main_function():
    iris_data_sets, iris_labels = utils.get_iris_data_set()
    iris_data_test_df, iris_data_train_df = utils.handle_irs_data(iris_data_sets, iris_labels)
    dt = DecisionTree()
    tree = dt.fit(iris_data_train_df)
    print(tree)
    print(iris_data_test_df)
    # r1 = dt.predict(["7.0", "3.2", "4.7", "1.4"])
    # print(r1)
    iris_test_result_df = predict_data(dt, iris_data_test_df, iris_labels)
    print(iris_test_result_df)


if __name__ == '__main__':
    main_function()
