from gensim.models.doc2vec import Doc2Vec
import gensim.downloader as api
import smart_open
import os
import gensim

# Set file names for train and test data
#test_data_dir = os.path.join(gensim.__path__[0], 'test', 'test_data')
#lee_train_file = os.path.join(test_data_dir, 'lee_background.cor')
#lee_test_file = os.path.join(test_data_dir, 'lee.cor')

lee_train_file = 'clean_text.txt'
def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

train_corpus = list(read_corpus(lee_train_file))
#test_corpus = list(read_corpus(lee_test_file, tokens_only=True))



model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
