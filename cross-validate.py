import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from useful_methods import *
import os
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

start_time = time.time()

parsed_path, prefix, choice = choose_dataset()
load_save_path = load_save_results(prefix, choice)

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(load_save_path, f'{prefix}_embeddings_word2vec.csv'))
    # df = pd.read_csv(os.path.join(load_save_path, f'{prefix}_embeddings_doc2vec.csv'))
    # df = pd.read_csv(os.path.join(load_save_path, f'{prefix}_embeddings_node2vec.csv'))
    # df = pd.read_csv(os.path.join(load_save_path, f'{prefix}_embeddings_graph2vec.csv'))

    X = df['embedding'].apply(lambda x: np.fromstring(x[1:-1], sep=',')).tolist()
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # cross validation for logistic regression

    log_reg = LogisticRegression(max_iter=1000)

    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'solver': ['liblinear', 'lbfgs', 'newton-cg', 'saga'],
        'penalty': ['l1', 'l2']
    }

    grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Parameters:", best_params)
    print("Best Score:", best_score)

    best_model = grid_search.best_estimator_
    test_accuracy = best_model.score(X_test_scaled, y_test)
    print("Test Set Accuracy with Best Parameters:", test_accuracy)

    # cross validation for MLP classifier

    # mlp_param_grid = {
    #     'hidden_layer_sizes': [(50,), (100,)],
    #     'activation': ['relu', 'tanh'],
    #     'solver': ['adam', 'sgd'],
    #     'alpha': [0.0001, 0.001, 0.01],
    #     'learning_rate': ['constant', 'adaptive']
    # }
    #
    # mlp = MLPClassifier(max_iter=1000)
    #
    # grid_search = GridSearchCV(mlp, mlp_param_grid, cv=5, scoring='accuracy')
    # grid_search.fit(X_train_scaled, y_train)
    #
    # best_params = grid_search.best_params_
    # best_score = grid_search.best_score_
    #
    # print("Best Parameters for MLPClassifier:", best_params)
    # print("Best Score for MLPClassifier:", best_score)
    #
    # best_model = grid_search.best_estimator_
    # test_accuracy = best_model.score(X_test_scaled, y_test)
    # print("Test Set Accuracy with Best Parameters for MLPClassifier:", test_accuracy)

    # cross validation for SVC classifier

    # svc_param_grid = {
    #     'C': [0.1, 1.0, 10.0],
    #     'kernel': ['linear', 'rbf'],
    #     'gamma': ['scale', 'auto']
    # }
    #
    # svc = SVC()
    #
    # grid_search = GridSearchCV(svc, svc_param_grid, cv=5, scoring='accuracy')
    # grid_search.fit(X_train_scaled, y_train)
    #
    # best_params = grid_search.best_params_
    # best_score = grid_search.best_score_
    #
    # print("Best Parameters for SVC:", best_params)
    # print("Best Score for SVC:", best_score)
    #
    # best_model = grid_search.best_estimator_
    # test_accuracy = best_model.score(X_test_scaled, y_test)
    # print("Test Set Accuracy with Best Parameters for SVC:", test_accuracy)
