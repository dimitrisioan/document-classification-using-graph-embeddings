import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


def choose_dataset():
    print("1 - 20 Newsgroups")
    print("2 - BBC Articles")
    print("3 - Emails (Spam and Ham)")

    while True:
        try:
            choice = int(input("Choose a dataset: "))
            if choice in [1, 2, 3]:
                break
            else:
                print("Invalid choice. Please select a valid option.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print("1 - basic preprocessing dataset")
    print("2 - basic preprocessing & stopwords removal dataset")
    print("3 - basic preprocessing & stemming/lemmatization  dataset")

    while True:
        try:
            sub_choice = int(input("Choose a specific parsed dataset: "))
            if sub_choice in [1, 2, 3]:
                break
            else:
                print("Invalid choice. Please select a valid option.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    if choice == 1:
        # 20newsgroups_path
        if sub_choice == 1:
            dataset_path = "datasets/20newsgroups/20newsgroups_parsed_1/"
        elif sub_choice == 2:
            dataset_path = "datasets/20newsgroups/20newsgroups_parsed_2/"
        elif sub_choice == 3:
            dataset_path = "datasets/20newsgroups/20newsgroups_parsed_3/"
        dataset_prefix = "20newsgroups"
    elif choice == 2:
        # bbc_path
        if sub_choice == 1:
            dataset_path = "datasets/bbc/bbc_parsed_1/"
        elif sub_choice == 2:
            dataset_path = "datasets/bbc/bbc_parsed_2/"
        elif sub_choice == 3:
            dataset_path = "datasets/bbc/bbc_parsed_3/"
        dataset_prefix = "bbc"
    elif choice == 3:
        # emails_path
        if sub_choice == 1:
            dataset_path = "datasets/emails/emails_parsed_1/"
        elif sub_choice == 2:
            dataset_path = "datasets/emails/emails_parsed_2/"
        elif sub_choice == 3:
            dataset_path = "datasets/emails/emails_parsed_3/"
        dataset_prefix = "emails"

    return dataset_path, dataset_prefix, sub_choice


def choose_dataset_for_word2vec():
    print("1 - 20 Newsgroups")
    print("2 - BBC Articles")
    print("3 - Emails (Spam and Ham)")

    while True:
        try:
            choice = int(input("Choose a dataset: "))
            if choice in [1, 2, 3]:
                break
            else:
                print("Invalid choice. Please select a valid option.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print("1 - basic preprocessing dataset")
    print("2 - basic preprocessing & stopwords removal dataset")
    print("3 - basic preprocessing & stemming/lemmatization  dataset")

    while True:
        try:
            sub_choice = int(input("Choose a specific parsed dataset: "))
            if sub_choice in [1, 2, 3]:
                break
            else:
                print("Invalid choice. Please select a valid option.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    if choice == 1:
        # 20newsgroups_path
        dataset_path = "datasets/20newsgroups/20newsgroups_dataset"
        dataset_prefix = "20newsgroups"
    elif choice == 2:
        # bbc_path
        dataset_path = "datasets/bbc/bbc_dataset"
        dataset_prefix = "bbc"
    elif choice == 3:
        # emails_path
        dataset_path = "datasets/emails/emails_dataset"
        dataset_prefix = "emails"

    return dataset_path, dataset_prefix, sub_choice


def load_save_results(dataset_prefix, sub_choice):
    if dataset_prefix == "20newsgroups":
        # 20newsgroups_path
        if sub_choice == 1:
            load_save_path = "results/20newsgroups_parsed_1"
        elif sub_choice == 2:
            load_save_path = "results/20newsgroups_parsed_2"
        elif sub_choice == 3:
            load_save_path = "results/20newsgroups_parsed_3"
        dataset_prefix = "20newsgroups"
    elif dataset_prefix == "bbc":
        # bbc_path
        if sub_choice == 1:
            load_save_path = "results/bbc_parsed_1"
        elif sub_choice == 2:
            load_save_path = "results/bbc_parsed_2"
        elif sub_choice == 3:
            load_save_path = "results/bbc_parsed_3"
        dataset_prefix = "bbc"
    elif dataset_prefix == "emails":
        # emails_path
        if sub_choice == 1:
            load_save_path = "results/emails_parsed_1"
        elif sub_choice == 2:
            load_save_path = "results/emails_parsed_2"
        elif sub_choice == 3:
            load_save_path = "results/emails_parsed_3"
        dataset_prefix = "emails"

    return load_save_path


def preprocess_file(file, remove_headers=False, remove_stopwords=False, lemmatization=False, stemming=False):
    with open(file, 'r') as f:
        text = f.read()
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


def file_to_sentences(file, remove_headers=False, remove_stopwords=False, lemmatization=False, stemming=False):
    sentences = []
    with open(file, 'r') as f:
        text = f.read()

    # Remove headers for 20newsgroups dataset
    lines = text.split('\n')
    if remove_headers:
        # Skip the first 3 lines (headers)
        lines = lines[3:]
    text = '\n'.join(lines)
    # print(text)

    raw_sentences = text.split('\n')
    for sentence in raw_sentences:
        # Remove urls and emails
        sentence = re.sub(r'http\S+|www\S+|\S+@\S+', '', sentence)
        # Remove punctuation
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        # Remove digits
        sentence = re.sub(r'\d', '', sentence)
        # Lowercase text
        sentence = sentence.lower()
        # tokenize text
        words = word_tokenize(sentence)
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
        if words:
            sentences.append(words)

    return sentences


def list_to_write(filename):
    a = read_file(filename)
    initial_list_of_words = a.split()
    # Remove every word until "Subject:" appears
    split_keyword = "Subject:"
    temp = initial_list_of_words.index(split_keyword)
    list_of_words = initial_list_of_words[temp + 1:]
    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in list_of_words]
    # Remove numbers
    list_of_words = [word for word in stripped if word.isalpha()]
    # Stopwords filtering
    stop_words = set(stopwords.words('english'))
    # print("the stopwords are %d" % len(stop_words))
    list_of_words = [word.lower() for word in list_of_words]
    list_of_words_size = len(list_of_words)
    # write_file(parsed_filepath, list_of_words)
    return list_of_words


def read_file(filename):
    # try:
    print("Loading %s ..." % filename)
    print("\n")
    # # Had encoding errors on the third file
    # # so I ignored characters that could not be decoded
    # fd = open(filename, errors="ignore")
    with open(filename, 'r') as f1:
        text = f1.read()
    return text


def write_file(filename, list_of_words_to_write):
    print('Parsing data on %s ...' % filename)
    with open(filename, 'w') as f2:
        for w in list_of_words_to_write:
            f2.write("%s\n" % w)
    return 0


def my_menu():
    print("[0] Exit")
    print("[1] Executing Word2Vec experiment...")
    print("[2] Executing Doc2Vec experiment...")
    print("[3] Executing Node2Vec experiment...")
    print("[4] Executing Graph2Vec experiment...")

    option = int(input("Enter your option: "))

    while option != 0:
        if option == 0:
            exit(0)
        if option == 1:
            print("[1] running")
        elif option == 2:
            print("[2] running")
        elif option == 3:
            print("[3] running")
        elif option == 4:
            print("[4] running")
        else:
            print("Invalid option.")
        print("\n \n")
        option = int(input("Enter your option: "))


def set_parameters_word2vec():
    return 0


def set_parameters_doc2vec():
    return 0


def set_parameters_node2vec():
    return 0


def set_parameters_graph2vec():
    return 0

