from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import smart_open
import gensim
from nltk.tokenize import word_tokenize
import time
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from ast import literal_eval
from sklearn.ensemble import RandomForestClassifier
from useful_methods import *

start_time = time.time()

parsed_path, prefix, choice = choose_dataset()
load_save_path = load_save_results(prefix, choice)

if __name__ == '__main__':

    # Load the Doc2Vec model from the corresponding dataset directory
    model = Doc2Vec.load(os.path.join(load_save_path, f'{prefix}_doc2vec'))

    # model = Doc2Vec.load("../document-classification-using-graph-embeddings/doc2vec_models/doc2vec.model")

    filecount = 0
    data = []
    X = []
    y = []
    # Loop through every subdirectory, read each text
    for category in os.listdir(parsed_path):
        category_path = os.path.join(parsed_path, category)
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            print(file_path)
            filecount += 1
            document_id = file.replace('.txt', '')
            tag = f"{document_id}_{category}"
            embedding = model.dv[tag]

            X.append(embedding)
            y.append(category)
    print(X[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train[0])
    print("-------------")
    print(y_train[0])
    # exit(0)
    # Train MLP classifier
    model_mlp = MLPClassifier()
    model_mlp.fit(X_train, y_train)

    # Predict the categories of the test data
    y_pred = model_mlp.predict(X_test)

    # Evaluate MLP classifier
    acc_score_mlp = accuracy_score(y_test, y_pred)
    prec_score_mlp = precision_score(y_test, y_pred, average='weighted')
    conf_matrix_mlp = confusion_matrix(y_test, y_pred)
    report_mlp = classification_report(y_test, y_pred)

    print("MLP classifier:\n")
    print(f"Accuracy: {acc_score_mlp}")
    print(f"Precision: {prec_score_mlp}")
    print(f"Confusion matrix:\n {conf_matrix_mlp}")
    print(f"Report:\n {report_mlp}")

    # Train RandomForest classifier
    model_rf = RandomForestClassifier()
    model_rf.fit(X_train, y_train)

    # Predict the categories of the test data
    y_pred_rf = model_rf.predict(X_test)

    # Evaluate RandomForest classifier
    acc_score_rf = accuracy_score(y_test, y_pred_rf)
    prec_score_rf = precision_score(y_test, y_pred_rf, average='weighted')
    conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
    report_rf = classification_report(y_test, y_pred_rf)

    print("RandomForest classifier:\n")
    print(f"Accuracy: {acc_score_rf}")
    print(f"Precision: {prec_score_rf}")
    print(f"Confusion matrix:\n {conf_matrix_rf}")
    print(f"Report:\n {report_rf}")

    # Train SVC classifier
    model_svc = SVC(kernel='linear', C=1.0, random_state=42)
    model_svc.fit(X_train, y_train)

    # Predict the categories of the test data
    y_pred_svc = model_svc.predict(X_test)

    # Evaluate SVC classifier
    acc_score_svc = accuracy_score(y_test, y_pred_svc)
    prec_score_svc = precision_score(y_test, y_pred_svc, average='weighted')
    conf_matrix_svc = confusion_matrix(y_test, y_pred_svc)
    report_svc = classification_report(y_test, y_pred_svc)

    print("SVC classifier:\n")
    print(f"Accuracy: {acc_score_svc}")
    print(f"Precision: {prec_score_svc}")
    print(f"Confusion matrix:\n {conf_matrix_svc}")
    print(f"Report:\n {report_svc}")

    # Train Logistic Regression classifier
    model_logreg = LogisticRegression(max_iter=1000)  # You can adjust max_iter as needed
    model_logreg.fit(X_train, y_train)

    # Predict the categories of the test data
    y_pred_logreg = model_logreg.predict(X_test)

    # Evaluate Logistic Regression classifier
    acc_score_logreg = accuracy_score(y_test, y_pred_logreg)
    prec_score_logreg = precision_score(y_test, y_pred_logreg, average='weighted')
    conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)
    report_logreg = classification_report(y_test, y_pred_logreg)

    print("Logistic Regression classifier:\n")
    print(f"Accuracy: {acc_score_logreg}")
    print(f"Precision: {prec_score_logreg}")
    print(f"Confusion matrix:\n {conf_matrix_logreg}")
    print(f"Report:\n {report_logreg}")

    print("--- %s seconds ---" % (time.time() - start_time))
