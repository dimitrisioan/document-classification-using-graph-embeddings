from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import smart_open
import gensim
from nltk.tokenize import word_tokenize
import time
import gensim.downloader as api
import os

start_time = time.time()

parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"

if __name__ == '__main__':

    models = {}
    filecount = 0
    documents = []

    # Loop through every subdirectory, read each word from every file
    for subdirectory in os.listdir(parsed_path):
        new_file_path = parsed_path + subdirectory + '/'
        for file in os.listdir(new_file_path):
            filecount += 1
            text_file = new_file_path + file
            print(text_file)
            with open(text_file, 'rt', errors="ignore") as f:
                text = f.read().split()
                document = TaggedDocument(words=text, tags=[file])
                documents.append(document)

    # Train a Doc2Vec model on the tokenized texts
    model = Doc2Vec(vector_size=300, window=5, min_count=5, epochs=50)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

    model.save("../document-classification-using-graph-embeddings/doc2vec_models/doc2vec.model")
    print(model.wv.most_similar('man', topn=10))

    print("Text files are:", filecount)

    print("--- %s seconds ---" % (time.time() - start_time))

# ALTERNATIVE SOLUTION
# Set file names for train and test data
# test_data_dir = os.path.join(gensim.__path__[0], 'test', 'test_data')
# lee_train_file = os.path.join(test_data_dir, 'lee_background.cor')
# lee_test_file = os.path.join(test_data_dir, 'lee.cor')

# train_file = 'clean_text.txt'
#
#
# # solution based on documentation Doc2Vec
# def read_corpus(fname, tokens_only=False):
#     with smart_open.open(fname, encoding="iso-8859-1") as f:
#         for i, line in enumerate(f):
#             tokens = gensim.utils.simple_preprocess(line)
#             if tokens_only:
#                 yield tokens
#             else:
#                 # For training data, add tags
#                 yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
#
#
# train_corpus = list(read_corpus(train_file))
# # test_corpus = list(read_corpus(lee_test_file, tokens_only=True))
#
#
# model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
# model.build_vocab(train_corpus)
# model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
#
# # alternative solution
#
# if __name__ == '__main__':
#     # Load data in text files
#     with open(train_file, 'r') as f:
#         each_text_data = f.read().replace('\n', ' ').split('subject:')
#
#     print(each_text_data)
#
#     # Tokenize data per text file
#     tokenized_data = []
#     for i, text in enumerate(each_text_data):
#         tokens = word_tokenize(text)
#         tagged_doc = TaggedDocument(words=tokens, tags=[i])
#         tokenized_data.append(tagged_doc)
#     print(tokenized_data)
#
#     # Train Doc2Vec model
#     model = Doc2Vec(tokenized_data, vector_size=100, window=5, min_count=1, workers=4)
#
#     # Get document vector
#     doc_vector = model.infer_vector(tokenized_data[0].words)
#     print(doc_vector)
