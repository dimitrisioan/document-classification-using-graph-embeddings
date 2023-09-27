import time
import os
import pandas as pd
import numpy
import networkx as nx
import matplotlib.pyplot as plt
from graph_creation_scripts import *
from k_core_modules import *
from karateclub import Graph2Vec
from sklearn.manifold import TSNE
from gensim import *

start_time = time.time()

dataset_path = "../document-classification-using-graph-embeddings/newsgroups_dataset/"
parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"
combined_parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_combined_parsed/"

# TODO IMPLEMENT DIFFERENTLY IT CONSUMES TOO MUCH RAM
# TODO ENSURE THAT EMBEDDINGS ARE SAVED FOR THE RIGHT TEXT_ID
if __name__ == '__main__':
    postingl = []
    all_data = []
    filecount = 0
    for subdirectory in os.listdir(parsed_path):
        directory = parsed_path + subdirectory + '/'
        data = []
        graphs = []
        # cnt = 0
        for file in os.listdir(directory):
            doc_file_path = directory + file
            print(doc_file_path)
            with open(doc_file_path, 'r', errors="ignore") as f:
                text = f.read().strip()

            # Check if the text has fewer than 3 words
            if len(text.split()) < 3:
                print(f"Skipping {file} due to insufficient words.")
                continue  # Skip to the next TXT file
            # to kanw gia kathe katigoria se ena ksexoristo csv kai ta enonw sto teliko

            unique_terms, term_freq, postingl, count_of_text = createInvertedIndexFromFile(doc_file_path, postingl)
            adjacency_matrix = CreateAdjMatrixFromInvIndex(unique_terms, term_freq)
            G = graphUsingAdjMatrix(adjacency_matrix, unique_terms)
            graphs.append(G)

            data.append({'document_id': file.replace('.txt', ''), 'embeddings': [], 'category': subdirectory})
            filecount += 1
            # cnt += 1
            # if cnt == 20:
            #     break

            # nx.draw(G, with_labels=True, node_color='skyblue', font_weight='bold')
            # plt.show()
            # print(unique_terms)
            # print(term_freq)
            # print(postingl)
            # print(count_of_text)
            # print(adjacency_matrix)
            # break
        # break

        # Graph2Vec training
        model = Graph2Vec()
        model.fit(graphs)
        graph_embeddings = model.get_embedding()

        # add embeddings for each document in data after training
        for i, embedding in enumerate(graph_embeddings):
            data[i]['embeddings'] = embedding.tolist()

        all_data.extend(data)

        print(f'Graph embeddings for {subdirectory} are: ', len(graph_embeddings))

        for _ in data:
            print(_)

        # Perform t-SNE dimensionality reduction
        tsne = TSNE(n_components=2)
        embeddings_2d = tsne.fit_transform(graph_embeddings)

        # Plot the embeddings
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        plt.title('Graph Embeddings Visualization (2D)')
        plt.show()

    # Create Dataframe and save data in a CSV file
    df = pd.DataFrame(all_data)
    df.to_csv('data_for_classifiers_graph2vec.csv', index=False)

    print("files are:", filecount)
    print("--- %s seconds ---" % (time.time() - start_time))
