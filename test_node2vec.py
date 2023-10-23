import sys
from gensim.models import Word2Vec
import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
from graph_creation_scripts import *
from k_core_modules import *
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle

from nodevectors import Node2Vec

start_time = time.time()

dataset_path = "datasets_2/20newsgroups/newsgroups_dataset/"
parsed_path = "datasets_2/20newsgroups/newsgroups_dataset_parsed/"

if __name__ == '__main__':
    # # -------------USING CORE RANK AND NODE2VEC ELIORC--------------
    # from node2vec import Node2Vec
    # newsgroups_dataset_4_categories = ['comp.windows.x',
    #                                    'rec.sport.baseball',
    #                                    'sci.space',
    #                                    'talk.religion.misc']
    # filecount = 0
    # cnt = 0
    # data = []
    # files_list = []
    # for category in os.listdir(parsed_path):
    #     if category not in newsgroups_dataset_4_categories:
    #         continue
    #     category_path = os.path.join(parsed_path, category)
    #     print(category_path)
    #     for file in os.listdir(category_path):
    #         file_path = os.path.join(category_path, file)
    #         print(file_path)
    #         filecount += 1
    #         cnt += 1
    #         files_list.append([file_path, os.path.getsize(file_path)])
    #         if cnt == 250:
    #             cnt = 0
    #             break
    # all_words = []
    # for document in files_list:
    #     document_path = document[0]
    #     with open(document_path, 'r', errors="ignore") as f:
    #         words = f.read().split()
    #         for word in words:
    #             if word not in all_words:
    #                 all_words.append(word)
    #
    # # Load union graph for 4 categories 1200 documents
    # # with open('union_graph_3.pkl', 'rb') as file:
    # #     union_graph = pickle.load(file)
    # with open('small_union_graph_3.pkl', 'rb') as file:
    #     union_graph = pickle.load(file)
    #
    # # Draw union graph
    # G = union_graph
    # print(nx.info(G))
    # # graphToPng(G)
    # # nx.draw(G)
    # # plt.show()
    #
    # # Get the core of union graph
    # G_core = nx.k_core(G, k=None, core_number=None)
    # print('---------------------------------------')
    # print('Core', nx.info(G_core))
    #
    # # Apply Node2Vec on core of the union graph
    # node2vec = Node2Vec(G_core, dimensions=64, walk_length=30, num_walks=10, workers=1)
    # model = node2vec.fit(window=10, min_count=1, batch_words=4)
    # model.wv.save_word2vec_format('node_embed_3.txt')
    #
    # print("files are:", filecount)
    # print(f' all_words = {all_words}')
    # print(f' all words are {len(all_words)}')

    # ---------------------------USING KARATE CLUB ------------------------

    from karateclub import Node2Vec

    with open('union_graph_3.pkl', 'rb') as file:
        union_graph = pickle.load(file)

    G = union_graph
    print(nx.info(G))

    unique_words = []
    for node in G.nodes():
        unique_words.append(node)

    # Create a mapping from integer labels to original string labels
    node_mapping = {i: node for i, node in enumerate(G.nodes())}

    converted_graph = nx.convert_node_labels_to_integers(G)
    print("Nodes:")
    print(list(converted_graph.nodes()))

    model = Node2Vec(walk_number=10, walk_length=30, dimensions=64, window_size=10, min_count=1)
    model.fit(converted_graph)
    numeric_node_embeddings = model.get_embedding()

    # Retrieve the original string node labels and their embeddings
    node_embeddings = {node_mapping[i]: emb for i, emb in enumerate(numeric_node_embeddings)}

    cnt = 0
    # Print the node embeddings for the original word nodes
    for word, embedding in node_embeddings.items():
        print(f"{word}: {embedding}")
        cnt += 1
        if cnt == 10:
            break


    filecount = 0
    cnt = 0
    data = []
    files_list = []

    for category in os.listdir(parsed_path):
        category_path = os.path.join(parsed_path, category)
        print(category_path)

        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            print(file_path)
            filecount += 1

            with open(file_path, "r", errors="ignore") as text_file:
                words = text_file.read().split()

                # Compute the embedding for the document
                words_found_vectors = [node_embeddings[word] for word in words if word in node_embeddings]

                result_embedding = np.sum(words_found_vectors, axis=0)

                # Normalize the embedding
                result_embedding = result_embedding / len(words_found_vectors)

                # Convert the embedding to a list
                result_embedding = np.array(result_embedding).tolist()

                document_id = file.replace('.txt', '')

                data.append({'document_id': document_id, 'embedding': result_embedding, 'category': category})

    # Create Dataframe and save data in a CSV file
    df = pd.DataFrame(data)
    # df.to_csv('data_for_classifiers_node2vec.csv', index=False)
    df.to_csv('data_for_classifiers_node2vec_karate_club.csv', index=False)

    exit(0)

    # ---------------------------USING FASTNODE2VEC------------------------
    # from fastnode2vec import Graph, Node2Vec
    # import networkx as nx
    #
    # with open('union_graph_3.pkl', 'rb') as file:
    #     union_graph = pickle.load(file)
    #
    # # Draw union graph
    # G = union_graph
    # print(nx.info(G))
    # # graphToPng(G)
    # # nx.draw(G)
    # # plt.show()
    #
    # edges = [(u, v) for u, v in G.edges()]
    #
    # # Create a fastnode2vec.Graph from the list of edges
    # graph = Graph(edges, directed=False, weighted=False)
    #
    # # Now, you can create a Node2Vec object and train it
    # n2v = Node2Vec(graph, dim=64, walk_length=30, window=10, p=2.0, q=0.5, workers=4)
    #
    # # Train the Node2Vec model
    # n2v.train(epochs=20)
    #
    # # # Access the word vectors for nodes
    # # print(n2v.wv["the"])
    #
    # # Create a dictionary with nodes as keys and embeddings as values
    # node2vec_embeddings = {node: n2v.wv[node] for node in n2v.wv.index_to_key}
    #
    # # Now you have a dictionary with nodes as keys and their embeddings as values
    # print(type(node2vec_embeddings))
    #
    # for node, embedding in node2vec_embeddings.items():
    #     print(node)
    #     print(embedding)
    #     break
    #
    # X = []
    # y = []
    # filecount = 0
    # cnt = 0
    # data = []
    # files_list = []
    #
    # for category in os.listdir(parsed_path):
    #     category_path = os.path.join(parsed_path, category)
    #     print(category_path)
    #
    #     for file in os.listdir(category_path):
    #         file_path = os.path.join(category_path, file)
    #         print(file_path)
    #         filecount += 1
    #
    #         with open(file_path, "r", errors="ignore") as text_file:
    #             words = text_file.read().split()
    #
    #             # Compute the embedding for the document
    #             words_found_vectors = [node2vec_embeddings[word] for word in words if word in node2vec_embeddings]
    #
    #             result_embedding = np.sum(words_found_vectors, axis=0)
    #
    #             # Normalize the embedding
    #             result_embedding = result_embedding / len(words_found_vectors)
    #
    #             # Convert the embedding to a list
    #             result_embedding = np.array(result_embedding).tolist()
    #             X.append(result_embedding)
    #             y.append(category)

    # --------------Classifiers---------------

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train MLP classifier
    model_mlp = MLPClassifier()
    model_mlp.fit(X_train, y_train)
    print("here")
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

