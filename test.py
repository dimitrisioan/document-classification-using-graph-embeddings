import time
import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
import networkx as nx
from graph_creation_scripts import *
from k_core_modules import *
from karateclub import Graph2Vec
from sklearn.manifold import TSNE
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from fastnode2vec import Node2Vec, Graph
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from useful_methods import *
from visualization import *

parsed_path, prefix, choice = choose_dataset()
load_save_path = load_save_results(prefix, choice)

start_time = time.time()


def contains_special_characters(file_path):
    special_char_pattern = r'[!@#$%^&*()_+={}\[\]:;"\'<>,.?/\\|]'
    try:
        with open(file_path, 'r') as file:
            content = file.read()

            match = re.search(special_char_pattern, content)
            if match:
                print(content)
                return True
            else:
                return False
    except FileNotFoundError:
        return False


if __name__ == '__main__':
    # Check embeddings for empty records or other string errors
    df = pd.read_csv(os.path.join(load_save_path, f'{prefix}_embeddings_word2vec.csv'))
    # df = pd.read_csv(os.path.join(load_save_path, f'{prefix}_embeddings_node2vec.csv'))

    for index, row in df.iterrows():
        try:
            embedding = np.fromstring(row['embedding'][1:-1], sep=',')
        except Exception as e:
            print(f"Error in row {index}: {e}")
            print(row)
    exit(0)
    # ------------------------------------------------------------------------------------
    text2 = '''Newsgroup: talk.religion.misc
    Document_id: 84053
    From: bgarwood@heineken.tuc.nrao.edu (Bob Garwood)
    Subject: Re: Who's next?  Mormons and Jews?

    In article <1r7os6$hil@agate.berkeley.edu>, isaackuo@spam.berkeley.edu (Isaac Kuo) writes:
    |> In article <C5wIA1.4Hr@apollo.hp.com> goykhman@apollo.hp.com (Red Herring) writes:
    |> >    The FBI claims, on the basis of their intelligence reports,
    |> >    that BD's had no plans to commit suecide.  They, btw, had bugged the 
    |> >    place and were listening to BD's conversations till the very end.
    |> >
    |> >    Koresh's attorney claims that, based on some 30 hours he spent
    |> >    talking to his client and others in the compound, he saw no
    |> >    indication that BD's were contemplating suecide.
    |> >
    |> >    The survivors claim it was not a suecide.
    |> 
    |> It's not clear that more than one of the survivors made this claim.  It is
    |> clear that at least one of the survivors made the contradictory claim that
    |> BD members had started the fire.

    No, this is far from clear.  We only have the word of the FBI spokepeople that
    a survivor made this claim.  We have the contradictory word of the lawyers who
    spoke with the survivors individually that ALL of them agreed that they did
    NOT have a suicide pact and did not intentionally start the fire.  In the absense
    of any more evidence, I don't see how we can decide who to believe.
    Furthermore, its quite possible that there was no general suicide pact and that
    some small inner circle took it upon themselves to kill everyone else.
    With the state of the area now, we may never know what happened.

    |> 
    |> >    BD's were not contemplating suecide, and there is no reason 
    |> >    to believe they committed one.
    |> 
    |> No reason?  How about these two:
    |> 
    |> 1.  Some of the survivors claimed that BD members poured fuel along the
    |> 	corridors and set fire to it.  The speed at which the fire spread
    |> 	is not inconsistent with this claim.

    Again, we have only the word of the FBI on this claim.  The lawyers who
    have also talked to the survors deny that any of them are making that claim.

    |> 
    |> 2.  There was certainly a fire which killed most of the people in the compound.
    |> 	There is a very very good possibility that the FBI did not start this
    |> 	fire.  This is a good reason to believe that the BD's did.

    I will agree on your assessment as to the relative probabilities.  Its more likely
    that the BD's started the fire than did the FBI.  But there is currently NO
    way to decide what actually happened based on the publically available evidence
    (which is nearly none).

    |> 
    |> 3.  Even if the BD's were not contemplating suicide, it is very possible that
    |> 	David Koresh was convinced (and thus convinced the others) that this
    |> 	was not suicide.  It was the fulfilment of a profecy of some sort.
    |> 
    |> There are three possibilities other than the BD's self destruction:
    |> 
    |> A.  They are not dead, but escaped via bunker,etc.  From reports of the
    |> 	inadequacies of the tunnels and the bodies found, I would rate this
    |> 	as highly unlikely.
    |> 
    |> B.  The fire was started by an FBI accident.  This is possible, but it would be
    |> 	foolish of us to declare this outright until more evidence can back it.
    |> 	Sure, it's possible that the armored vehicle knocked down a lantern
    |> 	which started the fire (why was there a lit lantern in the middle of
    |> 	the day near the edge of the complex?).  It's anecdotal evidence that
    |> 	has been contradicted by other escapees.
    |> 
    |> C.  The fire was started on purpose by the FBI.  This has been suggested by
    |> 	some on the NET, and I would rate this possibility as utterly
    |> 	ludicrous.  This is what we in "sci.skeptic" would call an
    |> 	"extraordinary claim" and won't bother refuting unless someone gives
    |> 	any good evidence to back it up.

       D.   The fire was an started accidentally by the BDs.  I am truely amazed that
            I have heard (or read) of no one suggesting this possibility.
            With all the tear gas and the lack of electical power in the compound and
            the adults wearing gas masks, it had to have been chaotic inside.
            I can easily image someone leaving a lamp too close to something or
            accidentally dropping a lamp or knocking one over.  With the winds, it
            would have quickly gotten out of control.

    |> 
    |> So we are left with two reasonable possibilities.  That the fire was an FBI
    |> accident and that the fire was started by the BD.  I find the latter more
    |> likely based on the evidence I've seen so far.

       No, I think that D is also quite reasonable.  I personally can't really
    asses any relative probablities to either of these 3 probabilities although if
    forced to bet on the issue, I would probably take an accident (either FBI or
    BD) over intential setting of the fire).

       I would also like to add a comment related to the reports that bodies recovered
    had gunshot wounds.  The coroner was on the Today Show this morning and categorically
    denied that they've reach any such conclusions.  He pointed out that under intense
    heat, sufficient pressure builds up in the head that can cause it to explode and
    that this can look very much like a massive gunshot wound to the head which is
    quite consisted with te reports I've read and heard.

       In short, there's been almost no evidence corroborating any of the many
    scenarios as to what happened on Monday.  We should remain skeptical until
    more information is available.  

    |> -- 
    |> *Isaac Kuo (isaackuo@math.berkeley.edu)	*       ___
    |> *					* _____/_o_\_____
    |> *	Twinkle, twinkle, little .sig,	*(==(/_______\)==)
    |> *	Keep it less than 5 lines big.	* \==\/     \/==/

    -- 

    Bob Garwood

    '''
    # import string
    #
    #
    # def preprocess_file(text, remove_headers=False, remove_stopwords=False, lemmatization=False, stemming=False):
    #     # Remove headers for 20newsgroups dataset
    #     lines = text.split('\n')
    #     if remove_headers:
    #         # Skip the first 4 lines (headers)
    #         lines = lines[4:]
    #     text = '\n'.join(lines)
    #     # print(text)
    #     # Remove urls and emails
    #     text = re.sub(r'http\S+|www\S+|\S+@\S+', '', text)
    #     # Remove punctuation
    #     text = text.translate(str.maketrans('', '', string.punctuation))
    #     # Remove digits
    #     text = re.sub(r'\d', '', text)
    #     # Lowercase text
    #     text = text.lower()
    #     # tokenize text
    #     words = word_tokenize(text)
    #     # Remove stopwords
    #     if remove_stopwords:
    #         stop_words = set(stopwords.words('english'))
    #         words = [word for word in words if word.lower() not in stop_words]
    #     # Lemmatization
    #     if lemmatization:
    #         lemmatizer = WordNetLemmatizer()
    #         words = [lemmatizer.lemmatize(word) for word in words]
    #     # Stemming
    #     if stemming:
    #         stemmer = PorterStemmer()
    #         words = [stemmer.stem(word) for word in words]
    #
    #     return words
    #     tokens = preprocess_file(text2)
    # print(tokens)
    filecount = 0
    # Loop through every subdirectory, read each text
    # for category in os.listdir(parsed_path):
    #     category_path = os.path.join(parsed_path, category)
    #     for file in os.listdir(category_path):
    #         file_path = os.path.join(category_path, file)
    #         if contains_special_characters(file_path):
    #             print(f"The file {file_path} contains special characters.")
    all_words = []
    for category in os.listdir(parsed_path):
        category_path = os.path.join(parsed_path, category)
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            with open(file_path, 'r') as f:
                words = f.read().split()
                all_words.extend(words)

    unique_words = list(set(all_words))
    print(f"all words are {len(all_words)}")
    print(f"all unique words are {len(unique_words)}")

    print("--- %s seconds ---" % (time.time() - start_time))
