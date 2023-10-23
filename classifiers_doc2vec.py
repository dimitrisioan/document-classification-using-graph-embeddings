from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from gensim.models import Word2Vec
import numpy as np
from gensim.models.doc2vec import Doc2Vec
import time
import os
import pandas as pd
from useful_methods import *


parsed_path, prefix, choice = choose_dataset()
load_save_path = load_save_results(prefix, choice)

start_time = time.time()

if __name__ == '__main__':

    # Load the Doc2Vec model from the corresponding dataset directory
    model = Doc2Vec.load(os.path.join(load_save_path, f'{prefix}_doc2vec'))

    # model = Doc2Vec.load("../document-classification-using-graph-embeddings/doc2vec_models/doc2vec.model")

    filecount = 0
    data = []

    # Loop through every subdirectory, read each text
    for category in os.listdir(parsed_path):
        category_path = os.path.join(parsed_path, category)
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            print(file_path)

            with open(file_path, "r") as text_file:
                words = text_file.read().split()
                # print(words)
                # print(len(words))
                # Skip file if it has less than 3 words
                if len(words) < 3:
                    continue
                filecount += 1
                document_id = file.replace('.txt', '')
                tag = f"{document_id}_{category}"
                embedding = model.dv[tag]

                data.append({'document_id': document_id, 'embedding': embedding.tolist(), 'category': category})

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Save the CSV file for Doc2Vec to the corresponding dataset directory
    df.to_csv(os.path.join(load_save_path, f'{prefix}_embeddings_doc2vec.csv'), index=False)

    # df.to_csv("data_for_classifiers_doc2vec.csv", index=False)

    print("Text files are:", filecount)
    print("--- %s seconds ---" % (time.time() - start_time))