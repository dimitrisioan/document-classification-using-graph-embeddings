import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report

start_time = time.time()

parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"

if __name__ == "__main__":
    df = pd.read_csv('data_for_classifiers_word2vec.csv')
    # df = pd.read_csv('data_for_classifiers_word2vec_300_google.csv')

    # Convert the embeddings column from string to list of floats
    X = df['embedding'].apply(lambda x: np.fromstring(x[1:-1], sep=',')).tolist()
    y = df['category']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    print("--- %s seconds ---" % (time.time() - start_time))
