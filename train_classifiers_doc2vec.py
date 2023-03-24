import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

start_time = time.time()

parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"

if __name__ == "__main__":
    df = pd.read_csv('data_for_classifiers_doc2vec.txt', sep=';', header=None, names=['document_id', 'embeddings', 'category'])

    # convert the embeddings column from string to list of floats
    df['embeddings'] = df['embeddings'].apply(lambda x: np.fromstring(x[1:-1], sep=','))

    X = df['embeddings'].tolist()
    y = df['category']

    # split the data into training and test sets and train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, alpha=0.001)
    model.fit(X_train, y_train)

    # predict the categories of the test data
    y_pred = model.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)
    prec_score = precision_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc_score}")
    print(f"Precision: {prec_score}")
    print(f"Confusion matrix:\n {conf_matrix}")

    # train RandomForest classifier
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    # predict categories using RandomForest classifier
    y_pred_rf = rf_model.predict(X_test)

    # evaluate RandomForest classifier
    acc_score_rf = accuracy_score(y_test, y_pred_rf)
    prec_score_rf = precision_score(y_test, y_pred_rf, average='weighted')
    conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

    print("RandomForest classifier:\n")
    print(f"Accuracy: {acc_score_rf}")
    print(f"Precision: {prec_score_rf}")
    print(f"Confusion matrix:\n {conf_matrix_rf}")

    print("--- %s seconds ---" % (time.time() - start_time))
