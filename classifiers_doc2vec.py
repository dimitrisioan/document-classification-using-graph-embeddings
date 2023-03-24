from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
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

    model = Doc2Vec.load("../document-classification-using-graph-embeddings/doc2vec_models/doc2vec.model")

    # Loop through the directories and documents
    with open("data_for_classifiers_doc2vec.txt", 'w') as output_file:
        for dir_name in os.listdir(parsed_path):
            for doc_name in os.listdir(parsed_path + dir_name):
                with open(parsed_path + dir_name + "/" + doc_name, "rt", errors="ignore") as text_file:
                    doc_text = text_file.read().strip()
                    print(text_file)
                    filecount += 1
                vector = model.infer_vector(doc_text.split())

                # Write the document embedding and category to the output file
                doc_id = doc_name.split(".")[0]
                category = dir_name
                embedding_str = ", ".join(str(x) for x in vector)
                output_file.write(f"{doc_id};[{embedding_str}];{category}\n")

        print("Text files are:", filecount)

    print("--- %s seconds ---" % (time.time() - start_time))
