from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import smart_open
import gensim
from nltk.tokenize import word_tokenize
import time
import os

start_time = time.time()

parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"

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
            with open(file_path, 'r', errors="ignore") as f:
                text = f.read().split()
                document = TaggedDocument(words=text, tags=[category])
                print(document)
                documents.append(document)

    # Train Doc2Vec model
    model = Doc2Vec(vector_size=300, window=5, min_count=5, epochs=50)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

    model.save("../document-classification-using-graph-embeddings/doc2vec_models/doc2vec.model")
    print(model.wv.most_similar('man', topn=10))

    print("Text files are:", filecount)

    print("--- %s seconds ---" % (time.time() - start_time))
