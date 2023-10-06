from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from gensim.models import Word2Vec
import numpy as np
from gensim.models.doc2vec import Doc2Vec
import time
import os
import pandas as pd

# from classification_functs import *
# from plot_functs import *

start_time = time.time()

parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"

if __name__ == '__main__':

    model = Doc2Vec.load("../document-classification-using-graph-embeddings/doc2vec_models/doc2vec.model")

    filecount = 0
    data = []

    # Loop through every subdirectory, read each text
    for category in os.listdir(parsed_path):
        category_path = os.path.join(parsed_path, category)
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            print(file_path)
            filecount += 1
            with open(file_path, 'r', errors="ignore") as f:
                text = f.read().split()
                document_id = file.replace('.txt', '')
                embedding = model.infer_vector(text)
                data.append({'document_id': document_id, 'embedding': embedding.tolist(), 'category': category})

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv("data_for_classifiers_doc2vec.csv", index=False)

    print("Text files are:", filecount)

    print("--- %s seconds ---" % (time.time() - start_time))



    # data = []
    # # Loop through the directories and documents
    # with open("data_for_classifiers_doc2vec.txt", 'w') as output_file:
    #     for category in os.listdir(parsed_path):
    #         for file in os.listdir(parsed_path + category):
    #             with open(parsed_path + category + "/" + file, "r", errors="ignore") as text_file:
    #                 doc_text = text_file.read().strip()
    #                 print(text_file)
    #                 filecount += 1
    #             vector = model.infer_vector(doc_text.split())
    #
    #             # Write the document embedding and category to the output file
    #             # document_id = file.replace('.txt', '') + '_' + category
    #             document_id = file.replace('.txt', '')
    #             embedding_str = ", ".join(str(x) for x in vector)
    #             output_file.write(f"{document_id};[{embedding_str}];{category}\n")
    #
    #     print("Text files are:", filecount)

