import nltk
from nltk.corpus import stopwords
from os import walk
# nltk.data.path.append("E:/python-projects/nltk")

import os
import shutil
import string
from nltk.corpus import stopwords

# Create the following folders in your system!
dataset_path = "../document-classification-using-graph-embeddings/newsgroups_dataset/"
parsed_path = "../document-classification-using-graph-embeddings/newsgroups_dataset_parsed/"


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

    list_of_words = [word.upper() for word in list_of_words]
    list_of_words_size = len(list_of_words)
    # write_file(parsed_filepath, list_of_words)
    return list_of_words


def read_file(filename):
    # try:
    print("Loading %s ..." % filename)
    print("\n")
    # # Had encoding errors on the third file
    # # so i ignored characters that could not be decoded
    # fd = open(filename, errors="ignore")
    fd = open(filename, 'r')
    text = fd.read()
    fd.close()
    return text


def write_file(filename, list_of_words_to_write):
    print('Parsing data on %s ...' % filename)
    fd = open(filename, 'w')
    for w in list_of_words_to_write:
        fd.write("%s\n" % w)
    return 0


# Delete all files on 'newsgroup_dataset_parsed'
for f in os.listdir(parsed_path):
    file_path = os.path.join(parsed_path, f)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

# Create subdirectories under 'newsgroup_dataset_parsed'
for file in os.listdir(dataset_path):
    # print(file)
    dataset_filepath = dataset_path + file
    parsed_filepath = parsed_path + file

    # Create subdirectory to save all parsed txt files
    if os.path.isfile(dataset_filepath):
        # print(file)
        dirName = parsed_filepath.replace('.txt', '')
        try:
            # Create target Directory
            os.mkdir(dirName)
            print("Directory ", dirName, " Created ")
        except FileExistsError:
            print("Directory ", dirName, " already exists")

        # Open every file form 'newsgroup_dataset_parsed' and
        # ignore unicode character errors
        fd = open(dataset_filepath, 'rt', errors="ignore")
        # text = fd.readline()
        # line = fd.readline()
        # sline = line.split()

        # Loop through every file on dataset and split it into
        #  different text files when "Newsgroup:" keyword appears
        with fd as bigfile:
            reader = bigfile.read()
            for i, text_part in enumerate(reader.split("Newsgroup:")[1:]):
                words = text_part.split()
                # Get document_id and save every individual text as a
                # separate txt file with its document_id as filename
                document_id = words[2]
                with open(os.path.join(dirName, str(document_id) + ".txt"), 'w') as new_file:
                    new_file.write("Newsgroup:" + text_part)

filecount = 0
# Loop through every subdirectory, tokenize every txt file and re-save each file
for subdirectory in os.listdir(parsed_path):
    new_file_path = parsed_path + subdirectory + '/'
    for file in os.listdir(new_file_path):
        filecount += 1
        test_file_path = new_file_path + file
        tokens = list_to_write(test_file_path)
        write_file(test_file_path, tokens)
        # print(tokens)

print("Text files are:", filecount)
'''
/------------------------------------T0DO TASKS FOR text_parser.py------------------------------------/

(DONE) 1. Iterate through every file on 'newsgroup_dataset' split on 'Newsgroup' keyword into txt files with document_id
 as name
(DONE) 2. Save the txt files on a subdirectory created under 'newsgroup_dataset_parsed'
(DONE) 3. Call 'list_to_write()' in order to tokenize every parsed txt file
(DONE) 4. re-save every txt file on the right format:
-upper case letters
-one word on every line
-remove numbers
-remove special characters
-remove 'Newsgroup:','document_id:','From:' complete lines(first 3 lines) and keyword 'Subject:'
5. Check if i need to optimize the parsing and the words that i am saving
6. Clean code
'''
