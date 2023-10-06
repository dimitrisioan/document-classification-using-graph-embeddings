from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import time
import os

# from classification_functs import *
# from plot_functs import *

start_time = time.time()

parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"

if __name__ == '__main__':

    model = Word2Vec.load("../document-classification-using-graph-embeddings/word2vec_models/word2vec.model")

    filecount = 0
    data = []

    # Loop through every subdirectory, read each word from every file
    for category in os.listdir(parsed_path):
        category_path = os.path.join(parsed_path, category)
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            print(file_path)
            filecount += 1
            with open(file_path, "r", errors="ignore") as text_file:
                words = text_file.read().split()

                # Compute the embedding for the document
                words_found_vectors = [model.wv.get_vector(word) for word in words if word in model.wv.key_to_index]
                result_embedding = np.sum(words_found_vectors, axis=0)

                # Normalize the embedding
                result_embedding = result_embedding / len(words_found_vectors)

                # Convert the embedding to a list
                result_embedding = np.array(result_embedding).tolist()
                # print(result_embedding)
                document_id = file.replace('.txt', '')

                data.append({'document_id': document_id, 'embedding': result_embedding, 'category': category})

    # Create Dataframe and save data in a CSV file
    df = pd.DataFrame(data)
    df.to_csv("data_for_classifiers_word2vec.csv", index=False)

    print("Text files are:", filecount)

    print("--- %s seconds ---" % (time.time() - start_time))




    # model = Word2Vec.load("../document-classification-using-graph-embeddings/word2vec_models/sentences_word2vec.model")
    #
    # filecount = 0
    # # Loop through every subdirectory, read each word from every file
    # with open("data_for_classifiers_word2vec.txt", 'w') as output_file:
    #     for category in os.listdir(parsed_path):
    #         for file in os.listdir(parsed_path + category):
    #             with open(parsed_path + category + "/" + file, "r", errors="ignore") as text_file:
    #                 filecount += 1
    #                 print(text_file)
    #                 words = text_file.read().split()
    #                 # print(words)
    #                 words_found_vectors = [model.wv.get_vector(word) for word in words if word in model.wv.key_to_index]
    #                 result_embedding = np.sum(words_found_vectors, axis=0)
    #                 # normalization of embeddings
    #                 result_embedding = result_embedding/len(words_found_vectors)
    #                 # separates the embedding values by comma
    #                 result_embedding = np.array(result_embedding).tolist()
    #                 # print(result_embedding))
    #                 # print(word)
    #                 document_id = file.replace('.txt', '')
    #                 output_file.write(f"{document_id};{result_embedding};{category}\n")
    #             # this break is used to access 1 file for each category
    #             # break
    #     print("Text files are:", filecount)


