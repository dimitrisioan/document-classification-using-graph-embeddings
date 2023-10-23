import os
from sklearn.feature_extraction.text import TfidfVectorizer
from useful_methods import *

parsed_path, prefix, choice = choose_dataset()

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
