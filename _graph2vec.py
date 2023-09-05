import time
import os
import pandas as pd
import numpy
import networkx as nx
import matplotlib.pyplot as plt
from graph_creation_scripts import *
from k_core_modules import *
from karateclub import Graph2Vec

start_time = time.time()

dataset_path = "../document-classification-using-graph-embeddings/newsgroups_dataset/"
parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"
combined_parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_combined_parsed/"


# TODO IMPLEMENT DIFFERENTLY IT CONSUMES TOO MUCH RAM
# TODO ENSURE THAT EMBEDDINGS ARE SAVED FOR THE RIGHT TEXT_ID
if __name__ == '__main__':
    postingl = []
    data = []
    embeddings = []
    graphs = []
    cnt = 0
    for subdirectory in os.listdir(parsed_path):
        new_file_path = parsed_path + subdirectory + '/'

        for file in os.listdir(new_file_path):
            doc_file_path = new_file_path + file
            print(doc_file_path)
            with open(doc_file_path, 'r', errors="ignore") as f:
                text = f.read().strip()

            # Check if the text has fewer than 3 words
            if len(text.split()) < 3:
                print(f"Skipping {file} due to insufficient words.")
                continue  # Skip to the next TXT file

            unique_terms, term_freq, postingl, count_of_text = createInvertedIndexFromFile(doc_file_path, postingl)
            adjacency_matrix = CreateAdjMatrixFromInvIndex(unique_terms, term_freq)
            G = graphUsingAdjMatrix(adjacency_matrix, unique_terms)
            graphs.append(G)

            data.append({'id': file, 'category': subdirectory})
            cnt += 1

            # nx.draw(G, with_labels=True, node_color='skyblue', font_weight='bold')
            # plt.show()
            # print(unique_terms)
            # print(term_freq)
            # print(postingl)
            # print(count_of_text)
            # print(adjacency_matrix)
        #     break
        # break

    model = Graph2Vec()
    model.fit(graphs)
    embeddings = model.get_embedding()

    for i, embedding in enumerate(embeddings):
        data[i]['embedding'] = embedding

    df = pd.DataFrame(data)

    embedding_df = pd.DataFrame(embeddings)
    embedding_df.to_csv('embeddings.csv', index=False)

    print(embedding_df)
    print("files are:", cnt)
    print(data)
    print("--- %s seconds ---" % (time.time() - start_time))
