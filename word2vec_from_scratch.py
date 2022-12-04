import os
import numpy as np
from text_parser import read_file

parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"
directory_path = "../document-classification-using-graph-embeddings/"


def mapping(tokens):
    word_to_id = {}
    id_to_word = {}

    for i, token in enumerate(set(tokens)):
        word_to_id[token] = i
        id_to_word[i] = token
    return word_to_id, id_to_word


def concat(*iterables):
    for iterable in iterables:
        yield from iterable


def one_hot_encode(id, vocab_size):
    res = [0] * vocab_size
    res[id] = 1
    return res


# np.random.seed(42)


def generate_training_data(tokens, word_to_id, window):
    X = []
    y = []
    n_tokens = len(tokens)

    for i in range(n_tokens):
        idx = concat(
            range(max(0, i - window), i),
            range(i, min(n_tokens, i + window + 1))
        )
        for j in idx:
            if i == j:
                continue
            X.append(one_hot_encode(word_to_id[tokens[i]], len(word_to_id)))
            y.append(one_hot_encode(word_to_id[tokens[j]], len(word_to_id)))

    return np.asarray(X), np.asarray(y)


def init_network(vocab_size, n_embedding):
    model = {
        "w1": np.random.randn(vocab_size, n_embedding),
        "w2": np.random.randn(n_embedding, vocab_size)
    }
    return model


def forward_propagation(model, X, return_cache=True):
    if len(X) == 0:
        return -1

    cache = {}

    cache["a1"] = X @ model["w1"]
    cache["a2"] = cache["a1"] @ model["w2"]
    cache["z"] = softmax(cache["a2"])

    if not return_cache:
        return cache["z"]
    return cache


def backward_propagation(model, X, y, alpha):
    cache = forward_propagation(model, X)
    da2 = cache["z"] - y
    dw2 = cache["a1"].T @ da2
    da1 = da2 @ model["w2"].T
    dw1 = X.T @ da1
    assert (dw2.shape == model["w2"].shape)
    assert (dw1.shape == model["w1"].shape)
    model["w1"] -= alpha * dw1
    model["w2"] -= alpha * dw2
    return cross_entropy(cache["z"], y)


def cross_entropy(z, y):
    return - np.sum(np.log(z) * y)


def softmax(X):
    res = []
    for x in X:
        exp = np.exp(x)
        res.append(exp / exp.sum())
    return res


def get_embedding(model, word):
    one_hot = []
    try:
        idx = word_to_id[word]
        one_hot = one_hot_encode(idx, len(word_to_id))
    except KeyError:
        print("`word` not in corpus")

    return forward_propagation(model, one_hot)["a1"]


def word_2_vec():
    print("Applying word2vec algorithm...")


if __name__ == '__main__':
    # for subdirectory in os.listdir(parsed_path):
    #     new_file_path = parsed_path + subdirectory + '/'
    #     for file in os.listdir(new_file_path):
    #         test_file_path = new_file_path + file
    #         # converts every tokenized text file into a Sting List
    #         tokens = read_file(test_file_path).split()
    #         print(tokens)
    #         # creating half of the lexicon
    #         word_to_id, id_to_word = mapping(tokens)
    #         X, y = generate_training_data(tokens, word_to_id, 2)
    #         print("X shape is:")
    #         print(X.shape)
    #         print("y shape is:")
    #         print(y.shape)
    #         # we set embedding size of features to 300 as the default value
    #         model = init_network(len(word_to_id), 10)
    #         break
    test_file_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/comp.os.ms-windows" \
                     ".misc/8514.txt "
    # converts every tokenized text file into a Sting List
    tokens = read_file(test_file_path).split()
    print(tokens)
    # creating half of the lexicon
    word_to_id, id_to_word = mapping(tokens)
    X, y = generate_training_data(tokens, word_to_id, 2)
    print("X shape is:")
    print(X.shape)
    print("y shape is:")
    print(y.shape)
    # we set embedding size of features to 300 as the default value
    model = init_network(len(word_to_id), 10)
    # learning = one_hot_encode(word_to_id["learning"], len(word_to_id))
    # result = forward(model, ["learning"], return_cache=False)[0]
    for word in range(len(tokens)):
        embedding = get_embedding(model, word)
        print(word)
        print(embedding)
