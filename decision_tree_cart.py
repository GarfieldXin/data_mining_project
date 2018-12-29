from time import clock

import numpy as np
import pandas as pd
from math import log
import data_sets_utils as utils
import datetime


class Node:
    def __init__(self, root=True, label=None, feature_name=None, feature=None, feature_value=None):
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.feature_value = feature_value
        # self.tree = {}
        self.true_branch = {}
        self.false_branch = {}
        self.result = {'root': self.root, 'label': self.label, 'feature': self.feature, 'feature_value': self.feature_value, 'true_branch': self.true_branch,
                       'false_branch': self.false_branch}

    def __repr__(self):
        return '{}'.format(self.result)

    # def add_node(self, val, node):
    #     self.tree[val] = node

    def add_true_node(self, node):
        self.true_branch['true'] = node

    def add_false_node(self, node):
        self.false_branch['false'] = node

    def predict(self, features):
        try:
            if self.root is True:
                return self.label
            else:
                if features[self.feature] >= self.feature_value:
                    return self.true_branch['true'].predict(features)
                else:
                    return self.false_branch['false'].predict(features)
            # return self.tree[features[self.feature]].predict(features)
        except:
            return "Other"

    def prune(self, min_gain, notify=False):
        if not self.root:
            self.true_branch['true'].prune(min_gain, notify)
            self.false_branch['false'].prune(min_gain, notify)


class DecisionTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    def fit(self, train_data):
        _, y_train, features = train_data.iloc[:, :-1], train_data.iloc[:, -1], train_data.columns[:-1]
        # print(_)
        # print("train_data.iloc[:, -1]:")
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

    def prune(self, min_gain, notify=False):
        return self._tree.prune(min_gain, notify)

    def train(self, train_data):
        _, y_train, features = train_data.iloc[:, :-1], train_data.iloc[:, -1], train_data.columns[:-1]
        # Single Node Tree
        if len(y_train.value_counts()) == 1:
            return Node(root=True, label=y_train.iloc[0])
        # Single Node Tree. Return the Max as the node class Tag.
        if len(features) == 0:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        current_score = self.gini(train_data)
        best_gain = 0.0
        best_attribute = None
        best_set1 = None
        best_set2 = None

        for index in range(len(features)):
            feature = features[index]
            column_values = train_data[feature]
            # print(column_values)
            for value in column_values.values:
                # print(value)
                set1, set2 = self.divide_set(train_data, feature, value)
                p = float(len(set1)) / len(train_data)
                gain = current_score - p * self.gini(set1) - (1 - p) * self.gini(set2)
                # print(str(feature) + str(" ") + str(value) + str(" Gini:"))
                # print(gain)
                if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                    # print(value)
                    best_gain = gain
                    best_attribute = (index, value)
                    best_set1 = set1
                    best_set2 = set2

        # print(best_gain)
        # print(best_attribute)
        # print(best_set1)
        # print(best_set2)

        node_tree = Node(root=False, feature_name=features[best_attribute[0]], feature=best_attribute[0],
                         feature_value=best_attribute[1])

        if best_gain > 0:
            sub_tree_true = self.train(best_set1)
            node_tree.add_true_node(sub_tree_true)
            sub_tree_false = self.train(best_set2)
            node_tree.add_false_node(sub_tree_false)
        else:
            label = train_data.iloc[:, -1].value_counts().sort_values(ascending=False).index[0]
            return Node(root=True, label=label)
        return node_tree

    def gini(self, data):
        total = len(data)
        counts = self.unique_counts(data)
        counts_index = counts.index
        imp = 0.0
        for i in counts_index:
            p1 = float(counts[i]) / total
            for j in counts_index:
                if i == j:
                    continue
                p2 = float(counts[j]) / total
                imp += p1*p2
        # print(imp)
        return imp

    def divide_set(self, train_data, feature, value):
        if self.is_int_float(value):
            set1 = train_data.loc[train_data[feature] >= value]
            set2 = train_data.loc[train_data[feature] < value]
        else:
            set1 = train_data.loc[train_data[feature] == value]
            set2 = train_data.loc[train_data[feature] != value]
            # print(set1)
            # print(set2)
        return set1, set2

    @staticmethod
    def unique_counts(data):
        return data.iloc[:, -1].value_counts()

    @staticmethod
    def is_int_float(value):
        try:
            complex(value)  # for int, long, float and complex
        except ValueError:
            return False
        return True


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
    # print(predict_result)
    predict_result_df = pd.DataFrame(predict_result, columns=iris_labels)
    return predict_result_df


def iris_sets_process():
    print("---------------------Iris Data Sets(CART)------------------------")
    iris_data_sets, iris_labels = utils.get_iris_data_set()
    iris_data_test_df, iris_data_train_df = utils.handle_data(iris_data_sets, iris_labels)
    # iris_data_train_df = pd.DataFrame(iris_data_sets, columns=iris_labels)
    target_names = np.unique(iris_data_train_df.iloc[:, -1])
    # print(target_names)
    dt = DecisionTree()
    fit_begin_time = datetime.datetime.now()
    print("Begin Time: " + str(fit_begin_time))
    tree = dt.fit(iris_data_train_df)
    # print(tree)
    fit_end_time = datetime.datetime.now()
    print("End Time: " + str(fit_end_time))
    print("Training used Time: " + str(fit_end_time - fit_begin_time))
    # r1 = dt.predict(["7.0", "3.2", "4.7", "1.4"])
    # print(r1)
    iris_test_result_df = predict_data(dt, iris_data_test_df, iris_labels)
    # print(iris_test_result_df)
    report = utils.generate_report(iris_data_test_df, iris_test_result_df, target_names)
    print(report)


def healthy_sets_process():
    print("---------------------Healthy Older People Sets(CART)------------------------")
    healthy_data_sets, healthy_labels = utils.get_healthy_data_set()
    healthy_data_test_df, healthy_data_train_df = utils.handle_data(healthy_data_sets, healthy_labels)
    target_names = np.unique(healthy_data_train_df.iloc[:, -1])
    # print(target_names)
    dt = DecisionTree()
    fit_begin_time = datetime.datetime.now()
    print("Begin Time: " + str(fit_begin_time))
    tree = dt.fit(healthy_data_train_df)
    fit_end_time = datetime.datetime.now()
    print("End Time: " + str(fit_end_time))
    print("Training used Time: " + str(fit_end_time - fit_begin_time))
    healthy_test_result_df = predict_data(dt, healthy_data_test_df, healthy_labels)
    report = utils.generate_report(healthy_data_test_df, healthy_test_result_df, target_names)
    print(report)


def autism_sets_process():
    print("---------------------Autism Adult Data Sets(CART)------------------------")
    autism_data_sets, autism_labels = utils.get_autism_data_set()
    autism_data_test_df, autism_data_train_df = utils.handle_data(autism_data_sets, autism_labels)
    autism_labels = np.array(autism_data_train_df.columns)
    # print(autism_labels)
    target_names = np.unique(autism_data_train_df.iloc[:, -1])
    # print(target_names)
    dt = DecisionTree()
    fit_begin_time = datetime.datetime.now()
    print("Begin Time: " + str(fit_begin_time))
    tree = dt.fit(autism_data_train_df)
    # print(tree)
    # print(autism_data_test_df)
    fit_end_time = datetime.datetime.now()
    print("End Time: " + str(fit_end_time))
    print("Training used Time: " + str(fit_end_time - fit_begin_time))
    autism_test_result_df = predict_data(dt, autism_data_test_df, autism_labels)
    report = utils.generate_report(autism_data_test_df, autism_test_result_df, target_names)
    print(report)


# if __name__ == '__main__':
def dt_cart():
    iris_sets_process()
    autism_sets_process()
    healthy_sets_process()
