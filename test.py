import time
import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
import networkx as nx
from graph_creation_scripts import *
from k_core_modules import *
from karateclub import Graph2Vec
from sklearn.manifold import TSNE
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
import pickle
from fastnode2vec import Node2Vec, Graph
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from useful_methods import *
from visualization import *

parsed_path, prefix, choice = choose_dataset()
load_save_path = load_save_results(prefix, choice)

start_time = time.time()


def contains_special_characters(file_path):
    special_char_pattern = r'[!@#$%^&*()_+={}\[\]:;"\'<>,.?/\\|]'
    try:
        with open(file_path, 'r') as file:
            content = file.read()

            match = re.search(special_char_pattern, content)
            if match:
                print(content)
                return True
            else:
                return False
    except FileNotFoundError:
        return False


if __name__ == '__main__':
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", min_df=3)

    tokenized_corpus = []
    for category in os.listdir(parsed_path):
        category_path = os.path.join(parsed_path, category)
        category_tfidf_vectors = []

        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            print(file_path)

            with open(file_path, "r") as text_file:
                words = text_file.read().split()
                # Skip file if it has less than 3 words
                if len(words) < 3:
                    continue
                tokenized_corpus.append(words)

    corpus = [' '.join(tokens) for tokens in tokenized_corpus]
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    print(tfidf_matrix)

    # -------------------------------------------------------------------------------------------
    stop_words = set(stopwords.words('english'))
    print(len(stop_words))
    # words = [word for word in words if word.lower() not in stop_words]
    # exit(0)

    # Check embeddings for empty records or other string errors
    df = pd.read_csv(os.path.join(load_save_path, f'{prefix}_embeddings_word2vec.csv'))
    # df = pd.read_csv(os.path.join(load_save_path, f'{prefix}_embeddings_node2vec.csv'))

    for index, row in df.iterrows():
        try:
            embedding = np.fromstring(row['embedding'][1:-1], sep=',')
        except Exception as e:
            print(f"Error in row {index}: {e}")
            print(row)
    # exit(0)
    # ------------------------------------------------------------------------------------

    # Convert the embeddings column from string to list of floats
    X = df['embedding'].apply(lambda x: np.fromstring(x[1:-1], sep=',')).tolist()
    y = df['category']
    print(type(X))

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
    print(f"Report:\n {report_mlp}")

    # Plotting Confusion Matrix as a Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_mlp, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - MLP Classifier')
    plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))
