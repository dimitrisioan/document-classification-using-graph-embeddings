import os
import glob
import nltk
import string
import itertools
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
# from karateclub import Node2Vec
from node2vec import Node2Vec



# G = nx.karate_club_graph()  # load data
#
# clubs = []  # list to populate with labels
# for n in G.nodes:
#     c = G.nodes[n]['club']  # karate club name, can be either 'Officer' or 'Mr. Hi'
#     clubs.append(1 if c == 'Officer' else 0)
#
# pos = nx.spring_layout(G, seed=42) # To be able to recreate the graph layout
# nx.draw_networkx(G, pos=pos, node_color = clubs, cmap='coolwarm') # Plot the graph
# plt.show()
#
# node2vec = Node2Vec(G, dimensions=64, walk_length=30, workers=1)


txt_file = 'test_files/test_file0.txt'
# Read and parse the text from the TXT file
with open(txt_file, 'r', encoding='utf-8') as file:
    text = file.read().strip()
    print(text)

    # Check if the text has fewer than 3 words
    if len(text.split()) < 3:
        print(f"Skipping {txt_file} due to insufficient words.")
        # continue  # Skip to the next TXT file

