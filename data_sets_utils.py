import pandas as pd
import os
import numpy as np
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

DIVISION_SCOPE = 5


def get_iris_data_set():
    iris_data_sets = []
    with open('Iris_Data_Set/iris.data', 'r') as file:
        data = file.readlines()
        for line in data:
            line_obj = line.split(',')
            line_array = []
            for obj in line_obj:
                obj = obj.replace("\n", "")
                line_array.append(obj)
            iris_data_sets.append(line_array)
        # print(iris_data_sets)
    labels = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']
    return iris_data_sets, labels


def handle_data(data, labels):
    test_data = []
    train_data = []
    for i in range(len(data)):
        if i % DIVISION_SCOPE == 0:
            test_data.append(data[i])
        else:
            train_data.append(data[i])
    print("Train Sets size: " + str(len(train_data)))
    print("Test Sets size: " + str(len(test_data)))
    test_data_df = pd.DataFrame(test_data, columns=labels)
    train_data_df = pd.DataFrame(train_data, columns=labels)
    # Clean Data Frame
    a_d = train_data_df.duplicated()
    b_d = test_data_df.duplicated()
    # print("The Duplicated items in Train Sets size: " + str(len(a_d)))
    # print("The Duplicated items in Test Sets size: " + str(len(b_d)))
    a_n = train_data_df.isnull()
    b_n = test_data_df.isnull()
    # print("The Null items in Test Sets size: " + str(len(a_n)))
    # print("The Null items in Train Sets size: " + str(len(b_n)))
    train_data_df = train_data_df.drop_duplicates()
    test_data_df = test_data_df.drop_duplicates()
    train_data_df = train_data_df.dropna()
    test_data_df = test_data_df.dropna()
    print("Clean Done. The Train Data Sets size: " + str(len(train_data_df)))
    print("Clean Done. The Test Data Sets size: " + str(len(test_data_df)))
    return test_data_df, train_data_df


def get_healthy_data_set():
    path_1 = "Datasets_Healthy_Older_People/S1_Dataset"
    path_2 = "Datasets_Healthy_Older_People/S2_Dataset"
    files_in_path1 = os.listdir(path_1)
    files_in_path2 = os.listdir(path_2)
    healthy_data_set = []
    for i in range(len(files_in_path1) - 1):
    # for i in range(0):  # For Test
        file_name = files_in_path1[i]
        gender = file_name[-1]
        with open(path_1 + "/" + file_name) as file:
            data = file.readlines()
            for line in data:
                line_obj = line.split(',')
                line_array = [gender]
                for obj in line_obj:
                    obj = obj.replace("\n", "")
                    line_array.append(obj)
                healthy_data_set.append(line_array)
    for i in range(len(files_in_path2) - 1):
    # for i in range(1):  # For Test
        file_name = files_in_path2[i]
        gender = file_name[-1]
        with open(path_2 + "/" + file_name) as file:
            data = file.readlines()
            for line in data:
                line_obj = line.split(',')
                line_array = [gender]
                for obj in line_obj:
                    obj = obj.replace("\n", "")
                    line_array.append(obj)
                healthy_data_set.append(line_array)
    print(len(healthy_data_set))
    # print(healthy_data_set)
    sets_headers = ['gender', 'starting_time', 'frontal_g', 'vertical_g',
                    'lateral_g', 'antenna_id', 'RSSI', 'Phase', 'Frequency', 'activity_label']
    return healthy_data_set, sets_headers


def generate_report(true_df, pred_df, target_names):
    y_true = np.array(true_df.iloc[:, -1])
    y_pred = np.array(pred_df.iloc[:, -1])
    # print(y_true)
    # print(y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names)
    return report


# if __name__ == '__main__':
#     sets, labels = get_healthy_data_set()
#     test_df, train_df = handle_irs_data(sets, labels)
#     # df = pd.DataFrame(sets, columns=labels)
#     print(len(train_df))
#     print(len(test_df))
    # print(test_df)
    # print(train_df)


