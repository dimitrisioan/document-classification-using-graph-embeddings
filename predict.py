from gensim.models.doc2vec import Doc2Vec
import time
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import string
from useful_methods import *

parsed_path, prefix, choice = choose_dataset()
load_save_path = load_save_results(prefix, choice)

start_time = time.time()


def preprocess_text(text, remove_headers=False, remove_stopwords=False, lemmatization=False, stemming=False):
    # Remove headers for 20newsgroups dataset
    lines = text.split('\n')
    if remove_headers:
        # Skip the first 3 lines (headers)
        lines = lines[3:]
    text = '\n'.join(lines)
    # print(text)
    # Remove urls and emails
    text = re.sub(r'http\S+|www\S+|\S+@\S+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d', '', text)
    # Lowercase text
    text = text.lower()
    # tokenize text
    words = word_tokenize(text)
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.lower() not in stop_words]
    # Lemmatization
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    # Stemming
    if stemming:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]

    return words


if __name__ == '__main__':
    # Load the Doc2Vec model from the corresponding dataset directory
    model = Doc2Vec.load(os.path.join(load_save_path, f'{prefix}_doc2vec'))

    # model = Doc2Vec.load("../document-classification-using-graph-embeddings/doc2vec_models/doc2vec.model")

    df = pd.read_csv(os.path.join(load_save_path, f'{prefix}_embeddings_doc2vec.csv'))

    # Convert the embeddings column from string to list of floats
    X = df['embedding'].apply(lambda x: np.fromstring(x[1:-1], sep=',')).tolist()
    y = df['category'].tolist()
    print(type(X))
    print(type(y))

    print(X[0:3])
    print(y[0:3])
    # unique_categories = []
    # for cat in y:
    #     if cat not in unique_categories:
    #         unique_categories.append(cat)
    # for cat in unique_categories:
    #     print(cat)

    print(len(X))
    print(len(y))

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = SVC(kernel='linear', C=1.0, random_state=42)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'accuracy of SVC classifier is {accuracy}')

    #     document_to_predict = '''
    # Subject: re: god
    # help god praise pray to you christ jesus mary church!! god father priest christian atheist
    #
    # '''

    # document_to_predict = '''
    # Subject: re:
    # I think we have a possibility to win this game. but the candidate is the opponent.
    #  they are a pretty good defensive team, but attack has to be good to win we have to score
    # a lot of goals.Good luck team!
    #
    # '''

    #     document_to_predict='''
    #     Subject: spring records
    #
    # 	The Orioles' pitching staff again is having a fine exhibition season.
    # Four shutouts, low team ERA, (Well, I haven't gotten any baseball news since
    # March 14 but anyways) Could they contend, yes. Could they win it all?  Maybe.
    #
    # But for all those fans of teams with bad spring records, remember Earl
    # Weaver's first law of baseball (From his book on managing)
    #
    # No one gives a damn in July if you lost a game in March. :)
    #
    # BTW, anyone have any idea on the contenders for the O's fifth starter?
    # It's pretty much set that Sutcliffe, Mussina, McDonald and Rhodes are the
    # first four in the rotation.
    #
    # Here at Johns Hopkins University where the mascot is the Blue Jay :(,
    # their baseball team logo was the Toronto club's logo. Now it's a
    # anatomically correct blue jay. God, can't they think of an original idea?
    # It's even in the same pose as the baltimore oriole on the O's hats.
    # How many people realize that the bird is really called a baltimore oriole?
    # __________________________________________________________________________
    # |Admiral Steve C. Liu        Internet Address: admiral@jhunix.hcf.jhu.edu|
    # |"Committee for the Liberation and Intergration of Terrifying Organisms  |
    # |and their Rehabilitation Into Society" from Red Dwarf - "Polymorph"     |
    # |****The Bangles are the greatest female rock band that ever existed!****|
    # |   This sig has been brought to you by... Frungy! The Sport of Kings!   |
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

    document_to_predict = '''Subject: As fun as this postseason has been -- and how it already has an all-timer of an October 
    highlight with the the first 8-5-3 double play in playoff history that ended Game 2 of the Braves-Phillies 
    National League Division Series -- one thing we haven’t seen yet is a decisive win-or-go-home game for both 
    teams. Frankly, we’ve only had two series so far that weren’t a sweep! And now there is only one Division Series 
    left. It is perhaps not a coincidence that it’s the one many thought would be the tightest series heading in.'''

    preprocessed_document = preprocess_text(document_to_predict)
    print(preprocessed_document)

    new_doc_vector = model.infer_vector(preprocessed_document)

    new_doc_vector = new_doc_vector.reshape(1, -1)
    print(new_doc_vector)

    predicted_category = classifier.predict(new_doc_vector)

    print(f'This document belongs in category {predicted_category}')