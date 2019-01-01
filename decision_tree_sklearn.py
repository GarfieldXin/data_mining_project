from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from numpy import unique
from pandas import DataFrame
import data_sets_utils as utils
import datetime


def healthy_sets_process():
    print("---------------------Healthy Older People Sets(CART in sklearn)------------------------")
    healthy_data_sets, healthy_labels = utils.get_healthy_data_set()
    healthy_data_df = DataFrame(healthy_data_sets, columns=healthy_labels)
    print("All data sets size: " + str(len(healthy_data_df)))
    target_names = unique(healthy_data_df.iloc[:, -1])
    healthy_data_train_df, healthy_data_test_df = train_test_split(healthy_data_df, test_size=0.3)
    print("Train Sets size: " + str(len(healthy_data_train_df)))
    print("Test Sets size: " + str(len(healthy_data_test_df)))
    tree = DecisionTreeClassifier(criterion="gini")
    fit_begin_time = datetime.datetime.now()
    print("Begin Time: " + str(fit_begin_time))
    tree.fit(healthy_data_train_df.iloc[:, :-1], healthy_data_train_df.iloc[:, -1])
    fit_end_time = datetime.datetime.now()
    print("End Time: " + str(fit_end_time))
    print("Training used Time: " + str(fit_end_time - fit_begin_time))
    y_true = healthy_data_test_df.iloc[:, -1]
    y_pred = tree.predict(healthy_data_test_df.iloc[:, :-1])
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)


def dt_sklearn_cart():
    healthy_sets_process()


# if __name__ == '__main__':
#     dt_sklearn_cart()
