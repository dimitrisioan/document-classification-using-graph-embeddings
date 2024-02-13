from graph_creation_scripts import *
from k_core_modules import *
import sys
import os
import time
import pickle
from useful_methods import *

parsed_path, prefix, choice = choose_dataset()
load_save_path = load_save_results(prefix, choice)
# load_save_path = 'test_results'

dir_name = "figures/0 test 700 0.2 1 1 20 100 0.08 n"


# Written by Kalogeropoulos, Skamnelos, Stathopoulos

# deletes by re drawing the graph edges of the graph given a minimum weight !needs fix but dont work
# def pruneGraphbyWeight(aMatrix, termlist):
#     print('pruning the graph')
#     temp = Woutdegree(aMatrix)  # [list of weight sums of each node]
#     maxval = (sum(temp[0]) / (len(temp[0]) * len(temp[0])))  # avarage weight of node
#     # maxval = (maxval * (+0.5)) + maxval #average weight of
#     gr = nx.Graph()
#     for i in range(len(aMatrix)):
#         gr.add_node(i, term=termlist[i])
#         for j in range(len(aMatrix)):
#             if i > j:
#                 if aMatrix[i][j] >= S * maxval:  # S is the persentage of the allowed weight based on maxval
#                     # print("i = %d j=%d weight  = %d" % (i, j, aMatrix[i][j]))
#                     gr.add_edge(i, j, weight=aMatrix[i][j])
#                 elif maxval < 1:
#                     gr.add_edge(i, j, weight=1)  # used because we had implemented a precentange on average
#     return (gr)


# Written by Kalogeropoulos, Skamnelos, Stathopoulos
def uniongraph(terms, term_freq, adjmatrix, collection_terms, union_graph_termlist_id, union_gr, id,
               collection_term_freq, kcore, kcorebool):
    # print(adjmatrix)
    for i in range(
            len(adjmatrix)):  # auto douleuei dioti o adjacency matrix proerxetai apo ton admatrix tou pruned graph opws kai o kcore ara uparxei
        # tautisi twn diktwn (i) twn kombwn
        if i in kcore and kcorebool == True:
            h = 0.06
        else:
            h = 1
        #
        # print(h)
        if terms[i] not in collection_terms:
            collection_terms[terms[i]] = id
            union_graph_termlist_id.append(id)
            collection_term_freq.append(term_freq[i] * (term_freq[i] + 1) * 0.5 * 0.05)
            union_gr.add_node(terms[i], id=id)
            id += 1
        elif terms[i] in collection_terms:
            index = collection_terms[terms[i]]
            collection_term_freq[index] += term_freq[i] * (term_freq[i] + 1) * 0.5 * 0.05
        for j in range(len(adjmatrix)):
            if i > j:
                if adjmatrix[i][j] != 0:
                    if terms[j] not in collection_terms:
                        collection_terms[terms[j]] = id
                        union_graph_termlist_id.append(id)
                        collection_term_freq.append(term_freq[j] * (term_freq[j] + 1) * 0.5 * 0.05)
                        union_gr.add_node(terms[i], id=id)
                        id += 1
                    # print('kcorbool = %s and h = %d '%(str(kcorebool),h ))
                    if not union_gr.has_edge(terms[i], terms[j]):
                        union_gr.add_edge(terms[i], terms[j], weight=adjmatrix[i][j] * 0.05)
                        # print("Calculating adj[",i,"]",j,"]: ",adjmatrix[i][j] * h)
                    elif union_gr.has_edge(terms[i], terms[j]):
                        union_gr[terms[i]][terms[j]]['weight'] += adjmatrix[i][j] * 0.05
                        # print("Calculating adj[",i,"]",j,"]: ",adjmatrix[i][j] * h)

    return terms, adjmatrix, collection_terms, union_graph_termlist_id, union_gr, id, collection_term_freq


# Written by Kalogeropoulos, Skamnelos, Stathopoulos
def loadNodeEmbedict(filename):
    final_list = []
    f = open(filename, "r")
    for line in f:
        line = line.rstrip("\n")
        line_list = line.split(' ')
        final_list.append(line_list)
    mydict = {}

    for x in final_list:
        mydict[x[0]] = x[1:len(x)]
    # for i in range(1, len(final_list)):
    #    mydict[final_list[i][0]] = final_list[i][1:len(final_list[i])]

    mydict = ({k: list(map(float, mydict[k])) for k in mydict})

    return mydict


# Written by Kalogeropoulos, Skamnelos, Stathopoulos
# SKAMNELOS IS USING THIS VERSION OF runit() IT IS DIFFERENT THAN THE ORIGINAL GSB
def runIt(filename, ans, window_flag, window_size, sen_par_flag, embe_flag, vaz_flag, par_window_size, per_window,
          dot_split, file_sum_mat):
    print(filename)
    temp = createInvertedIndexFromFile(filename, postinglist)
    # temp[0] = terms | temp[1] = term freq | temp[2] = posting list | temp 3 = doc number of words

    #######TEST############
    # print(temp[0])
    # print('\n')
    # print(temp[1])
    # print('\n')
    # print(temp[2])
    # print('\n')
    # print(temp[3])
    # print('\n')

    if window_flag:
        if embe_flag:
            if vaz_flag:
                try:
                    file = 'ugVazEmbe.txt'
                    filename1 = os.path.join(dir_name, file).replace("\\", "/")
                    # print(filename1)
                    testdict = loadNodeEmbedict(filename1)
                    # print(testdict)
                    adjmat = CreateAdjMatrix_Vazirgiannis_implementation_wn(temp[0], filename, window_size, mydict,
                                                                            testdict)

                except MemoryError:
                    sizeof_err_matrix = sys.getsizeof(adjmat)
                    print(sizeof_err_matrix)
                    exit(-1)
            else:
                try:
                    file = 'ugWindowEmbe.txt'
                    filename1 = os.path.join(dir_name, file).replace("\\", "/")
                    # print(filename1)
                    testdict = loadNodeEmbedict(filename1)
                    # print(testdict)
                    adjmat = CreateAdjMatrixFromInvIndexWithWindow_wn(temp[0], filename, window_size, per_window,
                                                                      dot_split, mydict, testdict)
                except MemoryError:
                    sizeof_err_matrix = sys.getsizeof(adjmat)
                    print(sizeof_err_matrix)
                    exit(-1)
        else:
            if vaz_flag:
                try:
                    adjmat = CreateAdjMatrix_Vazirgiannis_implementation(temp[0], filename, window_size)
                except MemoryError:
                    sizeof_err_matrix = sys.getsizeof(adjmat)
                    print(sizeof_err_matrix)
                    exit(-1)
            else:
                try:
                    adjmat = CreateAdjMatrixFromInvIndexWithWindow(temp[0], filename, window_size, per_window,
                                                                   dot_split)
                except MemoryError:
                    sizeof_err_matrix = sys.getsizeof(adjmat)
                    print(sizeof_err_matrix)
                    exit(-1)
    elif sen_par_flag:
        if embe_flag:
            try:
                file = 'ugSenParEmbe.txt'
                filename1 = os.path.join(dir_name, file).replace("\\", "/")
                # print(filename1)
                testdict = loadNodeEmbedict(filename1)
                # print(testdict)
                adjmat = CreateAdjMatrixFromInvIndexWithSenParWindow_wn(temp[0], filename, window_size, par_window_size,
                                                                        dot_split, mydict, testdict)
            except MemoryError:
                sizeof_err_matrix = sys.getsizeof(adjmat)
                print(sizeof_err_matrix)
                exit(-1)
        else:
            try:
                adjmat = CreateAdjMatrixFromInvIndexWithSenParWindow(temp[0], filename, window_size, par_window_size,
                                                                     dot_split)
            except MemoryError:
                sizeof_err_matrix = sys.getsizeof(adjmat)
                print(sizeof_err_matrix)
                exit(-1)
    else:
        if embe_flag:
            try:
                file = 'ugEmbe.txt'
                filename1 = os.path.join(dir_name, file).replace("\\", "/")
                # print(filename1)
                testdict = loadNodeEmbedict(filename1)
                # print(testdict)
                adjmat = CreateAdjMatrixFromInvIndex_wn(temp[0], temp[1], mydict, testdict)
            except MemoryError:
                sizeof_err_matrix = sys.getsizeof(adjmat)
                print(sizeof_err_matrix)
                exit(-1)
        else:
            try:
                adjmat = CreateAdjMatrixFromInvIndex(temp[0], temp[1])
            except MemoryError:
                sizeof_err_matrix = sys.getsizeof(adjmat)
                print(sizeof_err_matrix)
                exit(-1)
    # if int(filename[9:]) in bucket_list[5]:
    #   file_sum_mat = calculateSummationMatrix(adjmat,filename,file_sum_mat,temp[0],window_size)
    #######################
    try:
        gr = graphUsingAdjMatrix(adjmat, temp[0])
        docinfo.append([filename, temp[3]])
    except MemoryError:
        sizeof_err_matrix = sys.getsizeof(adjmat)
        print(sizeof_err_matrix)
        exit(-1)
    with open('1docinfo.dat', 'a') as file_handler:
        file_handler.write('%s %s \n' % (filename, temp[3]))
    file_handler.close()
    # print("----------------Using networkx method:---------------")
    # calculate the difference between min and max similarity and use it to prune our graph
    kcore = nx.Graph()
    kcore_nodes = []
    prunedadjm = nx.to_numpy_array(kcore)
    #	corebool --> not used(Can change to apply union graph penalty or not)
    #	splitfiles --> Window based splitting methods will be used if true(Exists because we use GSB in our experiments which doesnt require splitting)
    #	sen_par_flag --> if true sentence paragraph method will be used
    #	dot_split --> if true we will split according to "." using nltk's tokenize with punkt
    #	window_size --> The size of window when we are splitting using constant windows. If it is equal to 0
    #			per_window will be used instead.(per_window is the percentage of the text we will use)
    #	per_window --> A number between 0-1. 0 is 0% of the text while 1 is 100%. It is used to calculate the
    #			window size when using file length percentage based splitting.
    #	par_window_size --> Only used when sen_par_flag is true. It is the window size that corresponds to the paragraph level
    #	invfilename --> filename of the inverted index that will be constructed.
    #
    # Flag Hierarchy:
    #
    #	splitfiles >> sen_par_flag >> dot_split >> corebool(doesnt do anyting yet)
    #
    #	if splitfiles is false then we use gsb
    #
    #	if sen_par_flag is true we use sentence/paragraph splitting
    #		if dot_split is true the sentence portion of the adjmatrix will be split according to "."
    #		if dot split is false the file will be split according to window size(the sentence part only)
    #			if window size is 0 then the sentence portion of the adjmatrix will be generated according to percentage of the file(based on per_window)
    #			if window size is a positive integer the sentence portion of the adjmatrix will be generated according to that integer(constant window splitting)
    #		For the paragraph part we will be using constant window splitting according to par_window_size.(always)
    #	if sen_par_flag is false then we use regular splitting accoding to the rest of the flags/values
    #		if dot_split is true then we use splitting according to "." using nltk and punkt
    #		if dot split is false the file will be split according to window size
    #			if window size is 0 then the file will be split according to percentage of the file(based on per_window)
    #			if window size is a positive integer the file will be split according to that integer(constant window splitting)
    #
    #
    #
    # With current flags:
    #
    #	X=1 --> penalty on union graph splitting using percentages
    #	X=2 --> GSB
    #	X=3 --> penalty on union graph splitting using percentages
    #	X=4 --> penalty on union graph splitting using "." ISSUE:IF nltk is not installed, BY PASS: At menu 2: input one of existing indexes
    #	X=5 --> penalty on union graph splitting using constant window size
    #	X=6 --> penalty on union graph splitting using sentence/paragraph windows
    #
    #   !This is where we add methods to improve the graph such as core/truss decomposition, pruning, methods for important nodes. !

    #   NOTE: this main uses penalty on union graph (see lines 65,70,77 - uniongraph function) to punish frequent edges.
    if ans == 1:
        # By creating new graph we can translate it easily to the respective adj matrix
        # without calculating each edge weight separtly. It returns a pruned GRAPH
        print("Calculating without maincore:")
        prunedadjm = adjmat
        kcore_nodes = []
        # stopwordsStats(kcore,temp[0],filename)
        # print(nx.number_of_nodes(kcore))
        # print(len(kcore))
        # print(kcore.degree())
    if ans == 3:
        print("Calculating without maincore:")
        prunedadjm = adjmat
        kcore_nodes = []
    if ans == 4:
        print("Calculating without maincore:")
        prunedadjm = adjmat
        kcore_nodes = []

    ###################TEST####################
    if ans == 5:
        print("Calculating without maincore:")
        prunedadjm = adjmat
        kcore_nodes = []

    if ans == 6:
        print("Calculating without maincore:")
        prunedadjm = adjmat
        kcore_nodes = []

    if ans == 7:
        print("Calculating without maincore:")
        prunedadjm = adjmat
        kcore_nodes = []

    if ans == 8:
        print("Calculating without maincore:")
        prunedadjm = adjmat
        kcore_nodes = []

    #########################################################

    if ans == 2:
        print("Calculating without maincore:")
        prunedadjm = adjmat
        kcore_nodes = []
    term_freq = temp[1]
    # print(term_freq)
    return adjmat, temp[0], gr, term_freq, kcore_nodes, prunedadjm, file_sum_mat  # adjacency matrix terms list , graph


if __name__ == '__main__':
    filecount = 0
    cnt = 0
    data = []
    files_list = []
    for category in os.listdir(parsed_path):
        category_path = os.path.join(parsed_path, category)
        print(category_path)
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            print(file_path)
            filecount += 1
            cnt += 1
            files_list.append([file_path, os.path.getsize(file_path)])

        #     # break
        # # break
    files_list = sorted(files_list, key=itemgetter(1), reverse=True)
    print('First 5 elements in files_list', files_list[0:5])
    print("files are:", filecount)
    print(f'files_list contains {len(files_list)} files')

    ################################## DICTIONARY CREATION ##################################################
    #
    # final_list = []
    #
    # f = open("w2vecEmbe.txt", "r")
    # for line in f:
    #     line = line.rstrip("\n")
    #     line_list = line.split(' ')
    #     final_list.append(line_list)
    #
    mydict = {}
    #
    # for x in final_list:
    #     mydict[x[0]] = x[1:len(x) - 1]
    #
    # mydict = ({k: list(map(float, mydict[k])) for k in mydict})
    # print('--------->preproccess took %f mins \n' % ((end - start) / 60))

    ######################## HASHING #####################################
    bucket_list = []
    # bucket_list = BucketHash(file_list)
    # print(bucket_list)
    ########################--------test-------------#####################

    union_graph_termlist_id = []  # id of each unique word in the collection
    id = 0  # index used to iterate  on every list
    collection_terms = {}  # unique terms in the collection as a dict for performance
    union_graph = nx.Graph()  # Union of each graph of every document
    collection_term_freq = []
    ############################################
    file_sum_mat = []  # file and summation matrix
    stopword_weight_mat = []
    ############################################
    sumtime = 0
    # '''prints the menu '''
    # menu = printmenu()
    # print(menu)
    # S = menu[2]
    # hargs = menu[1]
    # menu = menu[0]
    # #######TEST#######
    # if len(sys.argv) <= 8 and len(sys.argv) >= 5:
    #     X = sys.argv[4]
    # elif len(sys.argv) != 4:
    #     print("some error here")
    #     exit(-99)
    ###################
    menu = 1
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!if menu==1 implement
    if menu == 1:
        X = input(
            ' 1.create index using maincore \n 2.create index without considering maincore \n 3.Create index using Density method \n 4.Create index using CoreRank method\nX = ')

        if int(X) == 1:
            corebool = False
            splitfiles = False
            sen_par_flag = False
            embe_flag = False
            vaz_flag = False
            dot_split = False
            window_size = 0
            par_window_size = 0
            per_window = 0
            invfilename = 'invertedindex.dat'
        elif int(X) == 2:
            corebool = False
            splitfiles = False
            sen_par_flag = False
            embe_flag = True
            vaz_flag = False
            dot_split = False
            window_size = 0
            par_window_size = 0
            per_window = 0
            invfilename = 'NegMain.dat'
        elif int(X) == 3:
            corebool = False
            splitfiles = True
            sen_par_flag = False
            embe_flag = False
            vaz_flag = False
            dot_split = False
            window_size = 0  # int(sys.argv[5])
            # window_size = 10
            par_window_size = 0
            per_window = 0.1  # float(sys.argv[7])
            invfilename = 'PerSplit.dat'
        elif int(X) == 4:
            corebool = False
            splitfiles = True
            sen_par_flag = False
            embe_flag = True
            vaz_flag = False
            dot_split = False
            window_size = 0
            par_window_size = 0
            per_window = float(sys.argv[7])  # 0.1
            invfilename = 'DotSplit.dat'
        ########TEST##############
        elif int(X) == 5:
            sen_par_flag = True
            splitfiles = False
            embe_flag = False
            vaz_flag = False
            corebool = False
            dot_split = False
            window_size = int(sys.argv[5])
            par_window_size = int(sys.argv[6])
            per_window = 0
            invfilename = 'ConstantWindFile.dat'
        elif int(X) == 6:
            sen_par_flag = True
            splitfiles = False
            embe_flag = True
            vaz_flag = False
            corebool = False
            dot_split = False
            window_size = int(sys.argv[5])
            par_window_size = int(sys.argv[6])
            per_window = 0
            invfilename = 'SenParConWind.dat'
        elif int(X) == 7:
            splitfiles = True
            sen_par_flag = False
            embe_flag = False
            vaz_flag = True
            corebool = False
            dot_split = False
            par_window_size = 0
            window_size = int(sys.argv[5])
            per_window = 0  # float(sys.argv[7])
            invfilename = 'Vazirgiannis.dat'
        elif int(X) == 8:
            splitfiles = True
            sen_par_flag = False
            embe_flag = True
            vaz_flag = True
            corebool = False
            dot_split = False
            par_window_size = 0
            window_size = int(sys.argv[5])
            per_window = 0  # float(sys.argv[7])
            invfilename = 'VazirgiannisW2Vec.dat'
        ##########################

        un_start = time.time()
        start = time.time()
        remaining = len(files_list) + 1

        for name in files_list:
            remaining -= 1

            print(remaining)
            name = name[0]
            # print("=========================For file = %s==================== " % name)
            gr = runIt(name, int(X), splitfiles, window_size, sen_par_flag, embe_flag, vaz_flag, par_window_size,
                       per_window, dot_split, file_sum_mat)
            # write doc info to file
            adjmatrix = gr[0]
            terms = gr[1]
            graph = gr[2]
            term_freq = gr[3]
            maincore = gr[4]
            prunedadjm = gr[5]
            file_sum_mat = gr[6]
            # getGraphStats(graph,name,True, True)

            try:
                ug = uniongraph(terms, term_freq, adjmatrix, collection_terms, union_graph_termlist_id, union_graph, id,
                                collection_term_freq, maincore, kcorebool=corebool)

            except MemoryError:
                sizeofgraph = sys.getsizeof(union_graph.edge) + sys.getsizeof(union_graph.node)
                print(sizeofgraph)
                exit(-1)
            #######################
            collection_terms = ug[2]
            #######################
            id = ug[5]
            union_graph = ug[4]
            collection_term_freq = ug[6]

            un_end = time.time()
            sumtime += (un_end - un_start) / 60
            # print('time spent on union graph = %f with adj matrix size %d' % (((un_end - un_start) / 60), len(adjmatrix)))
            # print('elapsed time = %f' % sumtime)

            # if int(X) == 1:
            #     nx.write_edgelist(union_graph, "ug.txt")
            # elif int(X) == 3:
            #     nx.write_edgelist(union_graph, "ugWindow.txt")
            # elif int(X) == 5:
            #     nx.write_edgelist(union_graph, "ugSenPar.txt")
            # elif int(X) == 7:
            #     nx.write_edgelist(union_graph, "ugVaz.txt")

            del graph
            del adjmatrix
            del prunedadjm
            del maincore

        # stopword_weight_mat = calculateStopwordWeight(file_sum_mat, collection_terms)
        # stopwordsStats(stopword_weight_mat, collection_terms)
        ######################################
        print('****Union Graph stats********')
        print(nx.info(union_graph))
        # ---------------------------------- graph to weights -----------------------
        print("calculating Term weights")
        print('=======================')

        end = time.time()
        print('+++++++++++++++++++++++++++')
        print('TIME = %f MINS' % ((end - start) / 60))
        print('++++++++++++++++++++++++++++')

        if int(X) == 1:
            # Save the union_graph to a file inside the specified directory
            file_path = os.path.join(load_save_path, 'union_graph_1.pkl')
            with open(file_path, 'wb') as file:
                pickle.dump(union_graph, file)
        elif int(X) == 2:
            file_path = os.path.join(load_save_path, 'union_graph_2.pkl')
            with open(file_path, 'wb') as file:
                pickle.dump(union_graph, file)
        elif int(X) == 3:
            file_path = os.path.join(load_save_path, 'union_graph_3.pkl')
            with open(file_path, 'wb') as file:
                pickle.dump(union_graph, file)