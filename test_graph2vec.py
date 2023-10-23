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
from useful_methods import *

start_time = time.time()

parsed_path, prefix, choice = choose_dataset()
load_save_path = load_save_results(prefix, choice)

if __name__ == '__main__':
    filecount = 0
    graphs = []
    data = []
    postingl = []

    for category in os.listdir(parsed_path):
        category_path = os.path.join(parsed_path, category)

        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            print(file_path)
            filecount += 1

            with open(file_path, 'r') as f:
                text = f.read().split()
                # Skip file if it has less than 3 words
                if len(text) < 3:
                    continue

            # Convert text document into Graph using GSB
            unique_terms, term_freq, postingl, count_of_text = createInvertedIndexFromFile(file_path, postingl)
            adjacency_matrix = CreateAdjMatrixFromInvIndex(unique_terms, term_freq)
            G = graphUsingAdjMatrix(adjacency_matrix, unique_terms)
            graphs.append(G)

    model = Graph2Vec(epochs=100)
    model.fit(graphs)
    graph_embeddings = model.get_embedding()

    filecount = 0
    for category in os.listdir(parsed_path):
        category_path = os.path.join(parsed_path, category)

        for file in os.listdir(category_path):
            document_id = file.replace('.txt', '')
            embedding = graph_embeddings[filecount].tolist()
            category = category

            data_entry = {
                'document_id': document_id,
                'embedding': embedding,
                'category': category
            }

            data.append(data_entry)
            filecount += 1

    print("files are:", filecount)
    print("--- %s seconds ---" % (time.time() - start_time))
