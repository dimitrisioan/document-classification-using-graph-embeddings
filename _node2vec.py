from gensim.models import Word2Vec
import os
import time

from sklearn.manifold import TSNE

from graph_creation_scripts import *
from k_core_modules import *
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

start_time = time.time()

dataset_path = "../document-classification-using-graph-embeddings/newsgroups_dataset/"
parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"

all_text_file = 'all_words.txt'


def createInvertedIndexFromFile(file, postingl):
    with open(file, 'r') as fd:
        # list containing every word in text document
        text = fd.read().split()
        uninque_terms = []
        termFreq = []
        for term in text:
            if term not in uninque_terms:
                uninque_terms.append(term)
                termFreq.append(text.count(term))
            if term not in postingl:
                postingl.append(term)
                postingl.append([file, text.count(term)])
            else:
                existingtermindex = postingl.index(term)
                if file not in postingl[existingtermindex + 1]:
                    postingl[existingtermindex + 1].extend([file, text.count(term)])
    # print(len(uninque_terms))
    # print(termFreq)
    return (uninque_terms, termFreq, postingl, len(text))


def CreateAdjMatrixFromInvIndex(terms, tf):
    # print("Adj_Matrix = %d * %d " % (len(terms), len(tf)))
    rows = numpy.array(tf)
    row = numpy.transpose(rows.reshape(1, len(rows)))
    col = numpy.transpose(rows.reshape(len(rows), 1))
    adj_matrix = numpy.array(numpy.dot(row, col))  # xi*xj
    # fullsize = rows.size + row.size + col.size + adj_matrix.size
    # print(fullsize / 1024 / 1024)
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if i == j:
                adj_matrix[i][j] = rows[i] * (rows[i] + 1) * 0.5
    # print(adj_matrix)
    # numpy.savetxt('test.txt', adj_matrix,fmt='%10.10f', delimiter=',')
    del row, rows, col
    return (adj_matrix)


def graphUsingAdjMatrix(adjmatrix, termlist, *args, **kwargs):
    gr = nx.Graph()
    filename = kwargs.get('filename', None)
    if not filename:
        filename = 'Name not found!'  # used when i want to visualize graphs with name

    for i in range(0, len(adjmatrix)):
        gr.add_node(i, term=termlist[i])
        for j in range(len(adjmatrix)):
            if i > j:
                gr.add_edge(i, j, weight=adjmatrix[i][j])
    # graphToPng(gr,filename = filename)
    return gr


if __name__ == '__main__':
    postingl = []
    # with open(all_text_file, "r") as file:
    #     data = file.read()
    #     file_words = data.split()
    #     # unique words list
    #     unique_words = list(set(file_words))
    #     print(f"Words: {len(file_words)}")
    #     print(f"Unique words: {len(unique_words)}")
    filecount = 0
    data = []
    for subdirectory in os.listdir(parsed_path):
        subdirectory_path = os.path.join(parsed_path, subdirectory)
        for file in os.listdir(subdirectory_path):
            filecount += 1
            test_file_path = os.path.join(subdirectory_path, file)
            print(test_file_path)

            unique_terms, term_freq, postingl, count_of_text = createInvertedIndexFromFile(test_file_path, postingl)
            adjacency_matrix = CreateAdjMatrixFromInvIndex(unique_terms, term_freq)
            G = graphUsingAdjMatrix(adjacency_matrix, unique_terms)
            # nx.draw(G)
            # plt.show()

            # Node2Vec
            # node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4) # default
            node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=10, workers=1)
            model = node2vec.fit(window=10, min_count=1, batch_words=4)

            embeddings_list = [model.wv[term] for term in model.wv.index_to_key]
            data.append({'id': file, 'embedding': embeddings_list, 'category': subdirectory})

            # # Example: Calculate cosine similarity between two terms
            # term1_embedding = model.wv[model.wv.index_to_key.index('term1')]
            # term2_embedding = model.wv[model.wv.index_to_key.index('term2')]
            # similarity = cosine_similarity([term1_embedding], [term2_embedding])
            # print(similarity)

            # print(data)
            break
        # break

    # Convert the 'data' list to a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv('node2_vec_data.csv', index=False)

    print("files are:", filecount)
    print("--- %s seconds ---" % (time.time() - start_time))
