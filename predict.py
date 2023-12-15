from gensim.models.doc2vec import Doc2Vec
import time
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
import os
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import string
from useful_methods import *

parsed_path, prefix, choice = choose_dataset()
load_save_path = load_save_results(prefix, choice)

start_time = time.time()


def calculate_doc_embedding(model, preprocessed_text):
    # Calculate document embedding based on chosen model
    embeddings = []
    skipped_words = []

    if isinstance(model, Word2Vec) or isinstance(model, Doc2Vec):
        for word in preprocessed_text:
            try:
                if word in model.wv.key_to_index:
                    embeddings.append(model.wv[word])
                else:
                    skipped_words.append(word)
            except KeyError:
                skipped_words.append(word)

        if len(embeddings) > 0:
            document_embedding = sum(embeddings) / len(embeddings)
            document_embedding = document_embedding.reshape(1, -1)
        else:
            print('No word found in the document!')

        return document_embedding, skipped_words


if __name__ == '__main__':
    while True:
        try:
            choose_model = int(input('1 - Word2Vec\n'
                                     '2 - Doc2Vec\n'
                                     '3 - Node2Vec\n'
                                     'Choose a model: '))
            if choose_model in [1, 2, 3]:
                break
            else:
                print("Please enter a valid option (1, 2, or 3).")
        except ValueError:
            print("Please enter a valid number (1, 2, or 3).")

    if choose_model == 1:
        model = Word2Vec.load(os.path.join(load_save_path, f'{prefix}_word2vec'))
        df = pd.read_csv(os.path.join(load_save_path, f'{prefix}_embeddings_word2vec.csv'))
    elif choose_model == 2:
        model = Doc2Vec.load(os.path.join(load_save_path, f'{prefix}_doc2vec'))
        df = pd.read_csv(os.path.join(load_save_path, f'{prefix}_embeddings_doc2vec.csv'))
    elif choose_model == 3:
        model = Word2Vec.load(os.path.join(load_save_path, f'{prefix}_node2vec'))
        df = pd.read_csv(os.path.join(load_save_path, f'{prefix}_embeddings_node2vec.csv'))

    total_words_model = len(model.wv.key_to_index)
    print(f"Total words in model's vocabulary: {total_words_model}")
    # Convert the embeddings column from string to list of floats
    X = df['embedding'].apply(lambda x: np.fromstring(x[1:-1], sep=',')).tolist()
    y = df['category'].tolist()

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression classifier
    model_logreg = LogisticRegression(max_iter=1000)
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

    # Preprocess the new document text
    new_document = 'test_files/test_file.txt'
    if choice == 1:
        preprocessed_text = preprocess_file(new_document)
    elif choice == 2:
        preprocessed_text = preprocess_file(new_document, remove_stopwords=True)
    elif choice == 3:
        preprocessed_text = preprocess_file(new_document, stemming=True, lemmatization=True)

    # Word2Vec/ Node2Vec selected
    if choose_model == 1 or choose_model == 3:
        embeddings, skipped_words = calculate_doc_embedding(model, preprocessed_text)

        if len(embeddings) > 0:
            document_embedding = sum(embeddings) / len(embeddings)
            document_embedding = document_embedding.reshape(1, -1)
        else:
            print('No word found in the document!')
            document_embedding = []

        predicted_category = model_logreg.predict(document_embedding)
        print(f"Predicted category for the new document: {predicted_category}")

        if skipped_words:
            print("Words we didn't find in the vocabulary:")
            print(skipped_words)
        else:
            print('No word found in the document!')

    # Doc2Vec selected
    if choose_model == 2:
        choose_doc2vec_method = int(input('1 - Use infer_vector method \n'
                                          '2 - Use average method \n'
                                          'Choose Doc2Vec method: '))
        if choose_doc2vec_method == 1:
            document_embedding = model.infer_vector(preprocessed_text)
            document_embedding = document_embedding.reshape(1, -1)
            predicted_category = model_logreg.predict(document_embedding)
            print(f"Predicted category for the new document: {predicted_category}")

        if choose_doc2vec_method == 2:
            embeddings, skipped_words = calculate_doc_embedding(model, preprocessed_text)

            if len(embeddings) > 0:
                document_embedding = sum(embeddings) / len(embeddings)
                document_embedding = document_embedding.reshape(1, -1)
            else:
                print('No word found in the document!')

            predicted_category = model_logreg.predict(document_embedding)
            print(f"Predicted category for the new document: {predicted_category}")

            if skipped_words:
                print("Words we didn't find in the vocabulary:")
                print(skipped_words)
            else:
                print('No word found in the document!')

    print("--- %s seconds ---" % (time.time() - start_time))

