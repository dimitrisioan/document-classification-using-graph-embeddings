from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from gensim.models import Word2Vec
import numpy as np
import time
import os
from sklearn.metrics import accuracy_score

# from classification_functs import *
# from plot_functs import *

start_time = time.time()

parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"

if __name__ == '__main__':
    filecount = 0

    model = Word2Vec.load("../document-classification-using-graph-embeddings/word2vec_models/sentences_word2vec.model")
    # Loop through every subdirectory, read each word from every file
    with open("data_for_classifiers.txt", 'w') as output_file:
        for dir_name in os.listdir(parsed_path):
            for file in os.listdir(parsed_path + dir_name):
                with open(parsed_path + dir_name + "/" + file, "rt", errors="ignore") as text_file:
                    filecount += 1
                    print(text_file)
                    words = text_file.read().split()
                    # print(words)
                    words_found_vectors = [model.wv.get_vector(word) for word in words if word in model.wv.key_to_index]
                    result_embedding = np.sum(words_found_vectors, axis=0)
                    # normalization of embeddings
                    result_embedding = result_embedding/len(words_found_vectors)
                    # separates the embedding values by comma
                    result_embedding = np.array(result_embedding).tolist()
                    # print(result_embedding))
                    # print(word)
                    output_file.write(f"{file.replace('.txt', '')};{result_embedding};{dir_name}\n")
                # this break is used to access 1 file for each category
                # break
        print("Text files are:", filecount)

    print("--- %s seconds ---" % (time.time() - start_time))
