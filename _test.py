import os
import shutil
import string
import time
from nltk.corpus import stopwords
from gensim.models import Word2Vec

start_time = time.time()

# Create the following folders in your system!
dataset_path = "../document-classification-using-graph-embeddings/newsgroups_dataset/"
parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"
combined_parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_combined_parsed/"

# split data into txts like I did in sentences "Subject:" included
if __name__ == '__main__':
    names = ['red', 'white', 'black', 'yellow']
    for name in names:
        print(names.index(name), name)
    print(dir(Word2Vec))
    print("--- %s seconds ---" % (time.time() - start_time))
