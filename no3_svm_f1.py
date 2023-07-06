import pandas as pd

from common import read_csv_data3, show_roc, evaluate_model
from sklearn.model_selection import train_test_split
import numpy as np

# 486 company, use 11 features
#
# two classes:
#   high value: h-index > h_index_threshold
#    low value: h-index <= h_index_threshold

h_index_threshold = 40

# total 11 features available
feature_list = ['repo count', 'contributor count', 'repo diff', 'contributor diff', 'degree', 'betweenness',
                'closeness', 'eigenvector', 'clustering', 'effective size', 'constraint']

# just change this settings to test other feature combinations
features_to_use = ['repo count', 'contributor count', 'repo diff', 'contributor diff', 'degree', 'betweenness',
                   'closeness', 'eigenvector', 'clustering', 'effective size', 'constraint']

def print_recall(y_true, y_pred):
    # refs: https://blog.csdn.net/csdnliwenqi/article/details/120791004
    from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
    print('\nRecall:', recall_score(y_true, y_pred, average='binary'))
    print('Precision score:', precision_score(y_true, y_pred, average='binary'))
    print('F1-score:', f1_score(y_true, y_pred, average='binary'))


def train_model_svm(X, y):
    from sklearn import svm
    model = svm.SVC(kernel='linear', C=16, verbose=True, probability=True)
    model.fit(X, y)
    return model


def check_f1_score(model, X_test, y_test):
    evaluate_model(model, X_test, y_test)

    # output roc
    y_pred = model.predict(X_test)
    # y_pred_proba = model.predict_proba(X_test)
    # show_roc(y_test, y_pred_proba[:, 1])

    print_recall(y_test, y_pred)


def main():
    names, data = read_csv_data3("data/status_22-12-24_08_48.csv")
    print('\n---------------------------')
    print('company count:', len(names))
    print('---------------------------')

    # add index to the data at end
    indexes = np.array([x for x in range(len(names))]).reshape(len(names), 1)
    data = np.concatenate((data, indexes), 1)

    # 0        1           2                  3                4
    # h-index, repo count, contributor count, repo-count diff, contributor diff
    X = data[:, [1, 2, 3, 4]]  # feature: repo count, contributor count, repo-count diff, contributor diff
    y = data[:, [0]].reshape(len(X))  # h-index as label

    X = X.astype(np.float32)

    # add features from graph
    X = X.tolist()
    df = pd.read_csv('data/matrix_graph_features.csv')
    graph_features = df.values.tolist()

    # construct dict
    # key: company name, value: 7 features(degree, betweenness, closeness, eigenvector, clustering, effective size, constraint)
    graph_feature_dict = {}
    for row in graph_features:
        graph_feature_dict[row[0]] = row[1:]

    for idx, name in enumerate(names):
        if name not in graph_feature_dict:
            print('There is no graph feature map for company:', name)
            continue
        # add 7 features from the graph
        X[idx] += graph_feature_dict[name]

    X = np.nan_to_num(np.array(X))

    # only use features defined in features_to_use
    fea_index = []
    for feature_name in features_to_use:
        idx = feature_list.index(feature_name)
        fea_index.append(idx)
    X = X[:, fea_index]

    y = y > h_index_threshold  # convert h-index to two classes: True for high value company, False for low value company
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('use svm model:')
    model = train_model_svm(X_train, y_train)
    check_f1_score(model, X_test, y_test)


if __name__ == '__main__':
    main()
