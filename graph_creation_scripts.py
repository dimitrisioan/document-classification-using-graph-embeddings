import nltk
import numpy
import statistics



##############################################################################################################################
### PART 1: NIKOLAOS SKAMNELOS usefull graph functions:

#CREATES a graph using a fixed sized window, by spliting the original textfile into sub-textfiles
# Split File in smaller files according to window size.
# If window size is equal to zero the function calculates
# the window by taking into account the total length of the
# file. (minimum window = 5)
def splitFileConstantWindow(file, window, per_window):
    # Open the file and split it into words
    inputFile = open(file, 'r').read().split()
    num_of_words = len(inputFile)
    outputFile = []

    # If window is equal to zero get window according to length or if percentage window flag is true
    if window == 0:
        # print(per_window)
        window = int(num_of_words * per_window) + 1
        # print("Window Size: ", window)
        if window < 5:
            window = 5

    # Join words according to window
    for i in range(0, num_of_words, window):
        outputFile.append(' '.join(inputFile[i:i + window]))

    #print(outputFile)
    return outputFile



def CreateAdjMatrixFromInvIndexWithWindow(terms, file, window_size, per_window, dot_split):
    #print("Adj_Matrix = %d * %d " % (len(terms), len(tf)))
    #print(terms)
    adj_matrix = numpy.zeros(shape=(len(terms), len(terms)))
    tfi = 0
    if dot_split:
        inputFile = open(file, 'r').read()
        split_file = nltk.tokenize.sent_tokenize(inputFile, language='english')
    else:
        split_file = splitFileConstantWindow(file, window_size, per_window)
    for subfile in split_file:
        window_terms = subfile.split()
        for term in window_terms:
            #print("\n")
            #print(term)
            row_index = terms.index(term)
            #print("TERM:",row_index)
            for x in range(0, len(window_terms)):
                col_index = terms.index(window_terms[x])
                #print("Y TERM:",col_index)
                if col_index == row_index:
                    tfi += 1
                else:
                    adj_matrix[row_index][col_index] += 1
            adj_matrix[row_index][row_index] += tfi * (tfi + 1) / 2
            tfi = 0

    # pen_adj_mat = applyStopwordPenalty(adj_matrix, terms)
    #print(adj_matrix)
    #numpy.savetxt('test.txt', adj_matrix,fmt='%10.10f', delimiter=',')
    # fullsize = rows.size + row.size + col.size + adj_matrix.size
    # print(fullsize / 1024 / 1024)
    return (adj_matrix)

def CreateAdjMatrixFromInvIndexWithWindow_embe(terms, file, window_size, per_window, dot_split, dict):
    # print("Adj_Matrix = %d * %d " % (len(terms), len(tf)))
    # print(terms)
    adj_matrix = numpy.zeros(shape=(len(terms), len(terms)))
    tfi = 0
    #if dot_split:
    #    inputFile = open(file, 'r').read()
    #    split_file = nltk.tokenize.sent_tokenize(inputFile, language='english')
    #else:
    split_file = splitFileConstantWindow(file, window_size, per_window)
    for subfile in split_file:
        window_terms = subfile.split()
        for term in window_terms:
            # print("\n")
            # print(term)
            row_index = terms.index(term)
            # print("TERM:",row_index)
            for x in range(0, len(window_terms)):
                col_index = terms.index(window_terms[x])
                # print("Y TERM:",col_index)
                if col_index == row_index:
                    tfi += 1
                else:
                    adj_matrix[row_index][col_index] += 1
            adj_matrix[row_index][row_index] += tfi * (tfi + 1) / 2
            tfi = 0

    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if terms[i] in dict and terms[j] in dict:
                if i == j:
                   word = dict[terms[i]]
                   embe = statistics.fmean(word)
                   adj_matrix[i][j] = adj_matrix[i][j] * embe
                else:
                    word_a = dict[terms[i]]
                    word_b = dict[terms[j]]
                    embe = numpy.add(word_a, word_b)
                    embe = statistics.fmean(embe)
                    adj_matrix[i][j] = adj_matrix[i][j] * embe
            else:
                adj_matrix[i][j] = adj_matrix[i][j]
    #print(adj_matrix)
    #numpy.savetxt('test2.txt', adj_matrix,fmt='%10.10f', delimiter=',')
    return (adj_matrix)



def CreateAdjMatrixFromInvIndexWithWindow_wn(terms, file, window_size, per_window, dot_split, word_embe_dict, node_embe_dict):
    # print("Adj_Matrix = %d * %d " % (len(terms), len(tf)))
    # print(terms)
    adj_matrix = numpy.zeros(shape=(len(terms), len(terms)))
    tfi = 0
    if dot_split:
        inputFile = open(file, 'r').read()
        split_file = nltk.tokenize.sent_tokenize(inputFile, language='english')
    else:
        split_file = splitFileConstantWindow(file, window_size, per_window)
    for subfile in split_file:
        window_terms = subfile.split()
        for term in window_terms:
            # print("\n")
            # print(term)
            row_index = terms.index(term)
            # print("TERM:",row_index)
            for x in range(0, len(window_terms)):
                col_index = terms.index(window_terms[x])
                # print("Y TERM:",col_index)
                if col_index == row_index:
                    tfi += 1
                else:
                    adj_matrix[row_index][col_index] += 1
            adj_matrix[row_index][row_index] += tfi * (tfi + 1) / 2
            tfi = 0

    #numpy.savetxt('test2.txt', adj_matrix,fmt='%10.10f', delimiter=',')
    return WordAndNodeEmbeCombo(terms, adj_matrix, word_embe_dict, node_embe_dict)

def WordAndNodeEmbeCombo(terms, adj_matrix, word_embe_dict, node_embe_dict):
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if terms[i] in word_embe_dict and terms[j] in word_embe_dict:
                if i == j:
                    word = word_embe_dict[terms[i]]
                    word_embe = statistics.fmean(word)
                    if terms[i] in node_embe_dict:
                        word = node_embe_dict[terms[i]]
                        node_embe = statistics.fmean(word)
                        embe = (word_embe + node_embe) * 0.5
                        adj_matrix[i][j] = adj_matrix[i][j] * embe
                    else:
                        adj_matrix[i][j] = adj_matrix[i][j] * word_embe
                else:
                    word_a = word_embe_dict[terms[i]]
                    word_b = word_embe_dict[terms[j]]
                    word_embe = numpy.add(word_a, word_b)
                    word_embe = statistics.fmean(word_embe)
                    if terms[i] in node_embe_dict and terms[j] in node_embe_dict:
                        word_c = node_embe_dict[terms[i]]
                        word_d = node_embe_dict[terms[j]]
                        node_embe = numpy.add(word_c, word_d)
                        node_embe = statistics.fmean(node_embe)
                        embe = (word_embe + node_embe) * 0.5
                        adj_matrix[i][j] = adj_matrix[i][j] * embe
                    else:
                        adj_matrix[i][j] = adj_matrix[i][j] * word_embe
            else:
                adj_matrix[i][j] = adj_matrix[i][j]
    #numpy.savetxt('test2.txt', adj_matrix, fmt='%10.10f', delimiter=',')
    return (adj_matrix)


# The idea behind this function is to get two adjacency matrices(one for sentences and one for paragraphs)
# using CreateAdjMatrixFromInvIndexWithWindow and then combine them into a single matrix using two weight
# coefficients a and b, which will determine the importance of each matrix.
def CreateAdjMatrixFromInvIndexWithSenParWindow(terms, file, sen_window_size, par_window_size, dot_split):
    matrix_size = len(terms)

    # Create the matrices
    sen_adj_matrix = numpy.zeros(shape=(matrix_size, matrix_size,))
    par_adj_matrix = numpy.zeros(shape=(matrix_size, matrix_size))

    # Get the adjacency matrix for each window
    sen_adj_matrix = CreateAdjMatrixFromInvIndexWithWindow(terms, file, sen_window_size, 0, dot_split)
    par_adj_matrix = CreateAdjMatrixFromInvIndexWithWindow(terms, file, par_window_size, 0, dot_split)

    # Create the final Matrix
    final_adj_matrix = numpy.zeros(shape=(matrix_size, matrix_size))
    # Create coefficients a and b
    a = 1.0
    b = 0.05

    # Add the two matrices
    final_adj_matrix = [[a * sen_adj_matrix[r][c] + b * par_adj_matrix[r][c] for c in range(len(sen_adj_matrix[0]))] for
                        r in range(matrix_size)]
    #print(final_adj_matrix)
    #numpy.savetxt('test.txt', final_adj_matrix,fmt='%10.5f', delimiter=',')
    return final_adj_matrix

def CreateAdjMatrixFromInvIndexWithSenParWindow_embe(terms, file, sen_window_size, par_window_size, dot_split, dict):
    matrix_size = len(terms)

    # Create the matrices
    sen_adj_matrix = numpy.zeros(shape=(matrix_size, matrix_size))
    par_adj_matrix = numpy.zeros(shape=(matrix_size, matrix_size))

    # Get the adjacency matrix for each window
    sen_adj_matrix = CreateAdjMatrixFromInvIndexWithWindow(terms, file, sen_window_size, 0, dot_split)
    par_adj_matrix = CreateAdjMatrixFromInvIndexWithWindow(terms, file, par_window_size, 0, dot_split)

    # Create the final Matrix
    final_adj_matrix = numpy.zeros(shape=(matrix_size, matrix_size))

    # Create coefficients a and b
    a = 1.0
    b = 0.05

    # Add the two matrices
    final_adj_matrix = [[a * sen_adj_matrix[r][c] + b * par_adj_matrix[r][c] for c in range(len(sen_adj_matrix[0]))] for
                        r in range(matrix_size)]
    # print(final_adj_matrix)
    #
    for i in range(len(final_adj_matrix)):
        for j in range(len(final_adj_matrix)):
            if terms[i] in dict and terms[j] in dict:
                if i == j:
                    word = dict[terms[i]]
                    embe = statistics.fmean(word)
                    final_adj_matrix[i][j] = final_adj_matrix[i][j] * embe
                else:
                    word_a = dict[terms[i]]
                    word_b = dict[terms[j]]
                    embe = numpy.add(word_a, word_b)
                    embe = statistics.fmean(embe)
                    final_adj_matrix[i][j] = final_adj_matrix[i][j] * embe
            else:
                final_adj_matrix[i][j] = final_adj_matrix[i][j]
    #print(adj_matrix)
    #numpy.savetxt('test2.txt', adj_matrix,fmt='%10.10f', delimiter=',')
    return (final_adj_matrix)

def CreateAdjMatrixFromInvIndexWithSenParWindow_wn(terms, file, sen_window_size, par_window_size, dot_split, word_embe_dict,node_embe_dict):
    matrix_size = len(terms)

    # Create the matrices
    sen_adj_matrix = numpy.zeros(shape=(matrix_size, matrix_size,))
    par_adj_matrix = numpy.zeros(shape=(matrix_size, matrix_size))

    # Get the adjacency matrix for each window
    sen_adj_matrix = CreateAdjMatrixFromInvIndexWithWindow(terms, file, sen_window_size, 0, dot_split)
    par_adj_matrix = CreateAdjMatrixFromInvIndexWithWindow(terms, file, par_window_size, 0, dot_split)

    # Create the final Matrix
    final_adj_matrix = numpy.zeros(shape=(matrix_size, matrix_size))

    # Create coefficients a and b
    a = 1.0
    b = 0.05

    # Add the two matrices
    final_adj_matrix = [[a * sen_adj_matrix[r][c] + b * par_adj_matrix[r][c] for c in range(len(sen_adj_matrix[0]))] for
                        r in range(matrix_size)]
    # print(final_adj_matrix)
    #return final_adj_matrix

    return WordAndNodeEmbeCombo(terms, final_adj_matrix, word_embe_dict, node_embe_dict)


#here the graph is created using a overlapping sliding window as Graph of word dictates
def CreateAdjMatrix_Vazirgiannis_implementation(terms, file, window_size):
    # print("Adj_Matrix = %d * %d " % (len(terms), len(tf)))
    #print(terms)
    adj_matrix = numpy.zeros(shape=(len(terms),len(terms)))
    split_file = open(file, 'r').read().split() #splitFileConstantWindow(file, window_size)
    counter = 0
    for term in split_file:
        row_index = terms.index(term)
        for x in range(0,window_size):
            try:
                col_index = terms.index(split_file[counter + x])
                adj_matrix[row_index][col_index]+=1
            except IndexError:
                break
        counter+=1
        adj_matrix[row_index][row_index]-=1
    #numpy.savetxt('test.txt', adj_matrix, fmt='%10.5f', delimiter=',')
    return (adj_matrix)

def CreateAdjMatrix_Vazirgiannis_implementation_embe(terms, file, window_size, dict):
    # print("Adj_Matrix = %d * %d " % (len(terms), len(tf)))
    #print(terms)
    adj_matrix = numpy.zeros(shape=(len(terms),len(terms)))
    split_file = open(file, 'r').read().split()
    counter = 0
    for term in split_file:
        row_index = terms.index(term)
        for x in range(0,window_size):
            try:
                col_index = terms.index(split_file[counter + x])
                adj_matrix[row_index][col_index]+=1
            except IndexError:
                break
        counter+=1
        adj_matrix[row_index][row_index]-=1

    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if terms[i] in dict and terms[j] in dict:
                if i == j:
                    word = dict[terms[i]]
                    embe = statistics.fmean(word)
                    adj_matrix[i][j] = adj_matrix[i][j] * embe
                else:
                    word_a = dict[terms[i]]
                    word_b = dict[terms[j]]
                    embe = numpy.add(word_a, word_b)
                    embe = statistics.fmean(embe)
                    adj_matrix[i][j] = adj_matrix[i][j] * embe
            else:
                adj_matrix[i][j] = adj_matrix[i][j]
    # print(adj_matrix)
    # numpy.savetxt('test2.txt', adj_matrix,fmt='%10.10f', delimiter=',')
    return (adj_matrix)

def CreateAdjMatrix_Vazirgiannis_implementation_wn(terms, file, window_size, word_embe_dict, node_embe_dict):
    # print("Adj_Matrix = %d * %d " % (len(terms), len(tf)))
    #print(terms)
    adj_matrix = numpy.zeros(shape=(len(terms),len(terms)))
    split_file = open(file, 'r').read().split() #splitFileConstantWindow(file, window_size)
    counter = 0
    for term in split_file:
        row_index = terms.index(term)
        for x in range(0,window_size):
            try:
                col_index = terms.index(split_file[counter + x])
                adj_matrix[row_index][col_index]+=1
            except IndexError:
                break
        counter+=1
        adj_matrix[row_index][row_index]-=1

    #return (adj_matrix)

    return WordAndNodeEmbeCombo(terms, adj_matrix, word_embe_dict, node_embe_dict)


########################################################################################################################
######### PART 2:KALOGEROPOULOS graph creation proccess and usefull graph functions


def createInvertedIndexFromFile(file, postingl):
    with open(file, 'r') as fd:
        # list containing every word in text document
        text = fd.read().split()
        uninque_terms = []
        termFreq = []
        for term in text:
            if term not in uninque_terms:
                uninque_terms.append(term)
                termFreq.append(text.count(term))
            if term not in postingl:
                postingl.append(term)
                postingl.append([file, text.count(term)])
            else:
                existingtermindex = postingl.index(term)
                if file not in postingl[existingtermindex + 1]:
                    postingl[existingtermindex + 1].extend([file, text.count(term)])
    # print(len(uninque_terms))
    # print(termFreq)
    return (uninque_terms, termFreq, postingl, len(text))

    ###############################lemmas################################
    # Weight_of_edge(i,j) = No.occurencies_of_i * No.occurencies_of_j   #
    #####################################################################


# using as an input the terms and the term frequency it creates the adjacency matrix of the graph
# in the main diagon we have the Win of each node of the graph and by the sum of each colume
# except the element of the diagon  is the  Wout of each node
# For more info see LEMMA 1 and LEMMA 2 of P: A graph based extension for the Set-Based Model, A: Doukas-Makris
def CreateAdjMatrixFromInvIndex(terms, tf):
    #print("Adj_Matrix = %d * %d " % (len(terms), len(tf)))
    rows = numpy.array(tf)
    row = numpy.transpose(rows.reshape(1, len(rows)))
    col = numpy.transpose(rows.reshape(len(rows), 1))
    adj_matrix = numpy.array(numpy.dot(row, col)) #xi*xj
    #fullsize = rows.size + row.size + col.size + adj_matrix.size
    #print(fullsize / 1024 / 1024)
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if i == j:
                adj_matrix[i][j] = rows[i] * (rows[i] + 1) * 0.5
    #print(adj_matrix)
    #numpy.savetxt('test.txt', adj_matrix,fmt='%10.10f', delimiter=',')
    del row, rows, col
    return (adj_matrix)

    ################################################################################################
    # For each node we calculate the sum of the elements of the respective row or colum of its index#
    # as its degree                                                                                 #
    ################################################################################################

def CreateAdjMatrixFromInvIndex_embe(terms, tf, dict):
    #print("Adj_Matrix = %d * %d " % (len(terms), len(tf)))
    rows = numpy.array(tf)
    row = numpy.transpose(rows.reshape(1, len(rows)))
    col = numpy.transpose(rows.reshape(len(rows), 1))
    adj_matrix = numpy.array(numpy.dot(row, col), dtype=float)
    #fullsize = rows.size + row.size + col.size + adj_matrix.size
    #print(fullsize / 1024 / 1024)
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if terms[i] in dict and terms[j] in dict:
                if i == j:
                    adj_matrix[i][j] = rows[i] * (rows[i] + 1) * 0.5
                    word = dict[terms[i]]
                    embe = statistics.fmean(word)
                    adj_matrix[i][j] = adj_matrix[i][j] * embe
                else:
                    word_a = dict[terms[i]]
                    word_b = dict[terms[j]]
                    embe = numpy.add(word_a, word_b)
                    embe = statistics.fmean(embe)
                    adj_matrix[i][j] = adj_matrix[i][j] * embe
            else:
                if i == j:
                    adj_matrix[i][j] = rows[i] * (rows[i] + 1) * 0.5
    #print(adj_matrix)
    #numpy.savetxt('test.txt', adj_matrix,fmt='%10.10f', delimiter=',')
    del row, rows, col
    return (adj_matrix)



def CreateAdjMatrixFromInvIndex_wn(terms, tf, word_embe_dict, node_embe_dict):
    # print("Adj_Matrix = %d * %d " % (len(terms), len(tf)))
    rows = numpy.array(tf)
    row = numpy.transpose(rows.reshape(1, len(rows)))
    col = numpy.transpose(rows.reshape(len(rows), 1))
    adj_matrix = numpy.array(numpy.dot(row, col), dtype=float)
    #fullsize = rows.size + row.size + col.size + adj_matrix.size
    #print(fullsize / 1024 / 1024)
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if i == j:
                adj_matrix[i][j] = rows[i] * (rows[i] + 1) * 0.5

    del row, rows, col
    #return (adj_matrix)
    return WordAndNodeEmbeCombo(terms, adj_matrix, word_embe_dict, node_embe_dict)


# computes the degree of every node using adj matrix
def Woutdegree(mat):
    list_of_degrees = numpy.sum(mat, axis=0)
    list_of_degrees = numpy.asarray(list_of_degrees)
    id = []
    # print(list_of_degrees)
    # print(numpy.size(list_of_degrees))
    for k in range(numpy.size(list_of_degrees)):
        id.append(k)
        list_of_degrees[k] -= mat[k][k]
    list_of_degrees.tolist()
    return list_of_degrees, id


def sortByDegree(val):
    return val[0]

