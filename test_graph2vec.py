import time
import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from graph_creation_scripts import *
from k_core_modules import *
from karateclub import Graph2Vec
from useful_methods import *
from collections import defaultdict

parsed_path, prefix, choice = choose_dataset()
load_save_path = load_save_results(prefix, choice)

start_time = time.time()

if __name__ == '__main__':
    # ------------train graph2vec model on bbc_parsed_1 dataset and create csv--------------------

    # postingl = []
    # data = []
    # graphs = []
    # filecount = 0
    #
    # for category in os.listdir(parsed_path):
    #     category_path = os.path.join(parsed_path, category)
    #     for file in os.listdir(category_path):
    #         file_path = os.path.join(category_path, file)
    #         print(file_path)
    #
    #         with open(file_path, 'r') as f:
    #             text = f.read().split()
    #             # Skip file if it has less than 3 words
    #             if len(text) < 3:
    #                 continue
    #
    #         filecount += 1
    #         # Convert text document into Graph using GSB
    #         unique_terms, term_freq, postingl, count_of_text = createInvertedIndexFromFile(file_path, postingl)
    #         adjacency_matrix = CreateAdjMatrixFromInvIndex(unique_terms, term_freq)
    #         G = graphUsingAdjMatrix(adjacency_matrix, unique_terms)
    #         graphs.append(G)
    #
    #         document_id = file.replace('.txt', '')
    #         data.append({'document_id': document_id, 'embedding': [], 'category': category})
    #         filecount += 1
    #         # break
    #     # break
    # print(f"all items in list graphs are {len(graphs)}")
    #
    # # Graph2Vec training
    # model = Graph2Vec(min_count=1)
    # model.fit(graphs)
    # graph_embeddings = model.get_embedding()
    #
    # # Add embeddings for each document in data after training
    # for i, embedding in enumerate(graph_embeddings):
    #     data[i]['embedding'] = embedding.tolist()
    #
    # # Create Dataframe and save data in a CSV file
    # df = pd.DataFrame(data)
    #
    # df.to_csv('bbc_embeddings_graph2vec.csv', index=False)
    #
    # print("Text files are:", filecount)
    #
    # df = pd.read_csv('bbc_embeddings_graph2vec.csv')

    # df = pd.read_csv('all_categories.csv')

    # # ----------------- train test split manually --------------------------
    df = pd.read_csv('bbc_embeddings_graph2vec.csv')

    # Convert the embeddings column from string to list of floats
    df['embedding'] = df['embedding'].apply(lambda x: np.fromstring(x[1:-1], sep=','))
    grouped = df.groupby('category')
    X_train, X_test, y_train, y_test = [], [], [], []

    for category, group in grouped:
        # Get the last 50 docs for testing
        test_data = group.tail(50)

        # Get the remaining docs for training
        train_data = group[~group.index.isin(test_data.index)]

        X_train.extend(train_data['embedding'])
        X_test.extend(test_data['embedding'])
        y_train.extend(train_data['category'])
        y_test.extend(test_data['category'])

    print(f"X_train size: {len(X_train)}")
    print(f"X_test size: {len(X_test)}")
    print(f"y_train size: {len(y_train)}")
    print(f"y_test size: {len(y_test)}")
    # for embedding,category in zip(X_test[0:2], y_test[0:2]):
    #     print(embedding,category)

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

    print("--- %s seconds ---" % (time.time() - start_time))
