import multiprocessing
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import smart_open
import gensim
from nltk.tokenize import word_tokenize
import time
import os
from useful_methods import *

parsed_path, prefix, choice = choose_dataset()
load_save_path = load_save_results(prefix, choice)

start_time = time.time()

if __name__ == '__main__':
    filecount = 0
    documents = []

    # Loop through every subdirectory, read each text
    for category in os.listdir(parsed_path):
        category_path = os.path.join(parsed_path, category)
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            print(file_path)
            filecount += 1
            with open(file_path, 'r') as f:
                text = f.read().split()
                document_id = file.replace('.txt', '')
                tag = f"{document_id}_{category}"
                document = TaggedDocument(words=text, tags=[tag])
                documents.append(document)
                # print(document)

    print(documents)
    print(len(documents))
    # exit(0)
    # Get the number of available CPU cores
    num_cores = multiprocessing.cpu_count()

    # Train Doc2Vec model
    # # bbc and emails dataset
    # model = Doc2Vec(vector_size=300, window=5, min_count=5, workers=num_cores, epochs=10, dm=1)
    # 20newsgroups dataset
    model = Doc2Vec(vector_size=300, window=5, min_count=5, workers=num_cores, epochs=50, dm=1)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

    # Save the Doc2Vec model to the corresponding dataset directory
    model.save(os.path.join(load_save_path, f'{prefix}_doc2vec'))

    # model.save("../document-classification-using-graph-embeddings/doc2vec_models/doc2vec.model")
    print(model.wv.most_similar('man', topn=10))

    print("Text files are:", filecount)
    print("--- %s seconds ---" % (time.time() - start_time))
