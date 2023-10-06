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

newsgroups_dataset_4_categories = ['comp.windows.x',
                                   'rec.sport.baseball',
                                   'sci.space',
                                   'talk.religion.misc']
# TODO IMPLEMENT DIFFERENTLY IT CONSUMES TOO MUCH RAM
# TODO ENSURE THAT EMBEDDINGS ARE SAVED FOR THE RIGHT TEXT_ID

if __name__ == '__main__':
    postingl = []
    data = []
    graphs = []
    filecount = 0
    
    for category in os.listdir(parsed_path):
        if category not in newsgroups_dataset_4_categories:
            continue
        category_path = os.path.join(parsed_path, category)
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            print(file_path)
            filecount += 1
            with open(file_path, 'r', errors="ignore") as f:
                text = f.read().strip()

            # Check if the text has fewer than 3 words and skip it
            if len(text.split()) < 3:
                print(f"Skipping {file} due to insufficient words.")
                continue
            # Convert text document into Graph using GSB
            unique_terms, term_freq, postingl, count_of_text = createInvertedIndexFromFile(file_path, postingl)
            adjacency_matrix = CreateAdjMatrixFromInvIndex(unique_terms, term_freq)
            G = graphUsingAdjMatrix(adjacency_matrix, unique_terms)
            graphs.append(G)

            document_id = file.replace('.txt', '')
            data.append({'document_id': document_id, 'embedding': [], 'category': category})
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
    print(f"all items in list graphs are {len(graphs)}")
    # Graph2Vec training
    model = Graph2Vec()
    model.fit(graphs)
    graph_embeddings = model.get_embedding()

    # Add embeddings for each document in data after training
    for i, embedding in enumerate(graph_embeddings):
        data[i]['embedding'] = embedding.tolist()

        # for _ in data:
        #     print(_)

        # # Perform t-SNE dimensionality reduction
        # tsne = TSNE(n_components=2)
        # embeddings_2d = tsne.fit_transform(graph_embeddings)
        #
        # # Plot the embeddings
        # plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        # plt.title('Graph Embeddings Visualization (2D)')
        # plt.show()

    # Create Dataframe and save data in a CSV file
    df = pd.DataFrame(data)
    df.to_csv('4_categories_graph2vec.csv', index=False)

    print("files are:", filecount)
    print("--- %s seconds ---" % (time.time() - start_time))
