import pandas as pd

from common import read_csv_data3, show_roc
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

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


def my_grid_search_and_cv(clf, X_train, X_test, y_train, y_test, detailed_info):
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:", clf.best_params_);
    y_true, y_pred = y_test, clf.predict(X_test)
    y_true, y_pred_proba = y_test, clf.predict_proba(X_test)
    try:
        best_estimator_feature_importance = clf.best_estimator_.feature_importances_
        # print("feature importance:", clf.best_estimator_.feature_importances_)

        print("feature importance: ({} features in total)".format(len(features_to_use)))
        for x in range(0, len(features_to_use)):
            print(features_to_use[x], best_estimator_feature_importance[x])
    except:
        pass

    if (detailed_info == True):
        print("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print()

    # print(classification_report(y_true, y_pred, digits=3))
    #    print (y_pred, '++')
    #    print (y_pred_proba[:, 1], '++')
    # print('AUC:', roc_auc_score(y_test, y_pred_proba[:, 1]))
    print()
    return y_pred



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

    X = np.array(X)

    # only use features defined in features_to_use
    fea_index = []
    for feature_name in features_to_use:
        idx = feature_list.index(feature_name)
        fea_index.append(idx)
    X = X[:, fea_index]

    y = y > h_index_threshold  # convert h-index to two classes: True for high value company, False for low value company
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    random_seed = 42

    parameters_grid_XGBoost = {
        'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5],  # best: 0.3
        'n_estimators': range(80, 110, 10),  # best: 80
        'subsample': [0.8, 0.9, 1],  # best: 1
        'min_child_weight': [1, 3, 5, 7],  # best: 5
        # 'colsample_bylevel': np.linspace(0.6, 2, 5)
    }

    f1 = 'accuracy'
    grid_model = GridSearchCV(XGBClassifier(random_state=random_seed), parameters_grid_XGBoost, cv=5, scoring=f1,
                              n_jobs=1, verbose=1)
    y_pred_XGB = my_grid_search_and_cv(grid_model, X_train, X_test, y_train, y_test, False)
    print(y_pred_XGB)
    print('model score:', grid_model.score(X_test, y_test))

    # output roc
    model = grid_model.best_estimator_
    y_pred = model.predict(X_test, output_margin=True)
    show_roc(y_test, y_pred)

    print_recall(y_test, y_pred_XGB)


if __name__ == '__main__':
    main()
