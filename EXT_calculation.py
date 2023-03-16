"""""""""""""""""""""""""""""""""""""""""""""
In this file we calculate the children's
networks' EXT values.

EXT stands for an externally driven network growth scenario,
and has also been referred to as PAQ = Preferential Acquisition.
- an input based model of acquisition
- Acquisition of new nodes based on connectiveness
    of similar nodes
- Prediction: The early vocabulary will
    reflect the input
- EXT-values: predicts that words that connect
    to more words in the existing network
    will be learned.
- the value of a new node is its degree in
    its learning environment
"""""""""""""""""""""""""""""""""""""""""""""
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

#   import needed files
LD = pd.read_csv('.\csv_files\LD_norm.csv', sep=",", header=int(), index_col=0)
FDL = pd.read_csv('.\csv_files\FDL_norm_eucl.csv', sep=",", header=int(), index_col=0)
FDK = pd.read_csv('.\csv_files\FDK_norm_eucl.csv', sep=",", header=int(), index_col=0)

#   where to save outputs
save_to_csv = './csv_files/'

"""""""""""""""""""""""""""
create final network
"""""""""""""""""""""""""""

#   choose distance measure
Lev_dist = True
FDL_dist = True
FDK_dist = True


THRESHOLD = False
WEIGHTED = True


"""
Now we calculate INT-values which are based on edges which only exist if the dist between two words
is smaller or equal 0.25.
"""

if THRESHOLD:
    for dist_type, dist_type_str in zip([LD, FDL, FDK], ['LD', 'FDL', 'FDK']):
        G = nx.Graph()  # creates an empty graph
        #   add nodes
        nodes = dist_type.columns
        G.add_nodes_from(nodes)

        #   add edges
        edges = []
        for word_1 in nodes:
            for word_2 in nodes:
                if word_1 is not word_2:
                    #   if words are in the df multiple times, we choose the first occurrence
                    if dist_type.at[word_1, word_2].shape == (2,):
                        if dist_type.at[word_1, word_2][0] <= 0.25:
                            edges.append((word_1, word_2))
                    else:
                        if dist_type.at[word_1, word_2] <= 0.25:
                            edges.append((word_1, word_2))
        G.add_edges_from(edges)
        #   draw graph
        nx.draw(G)
        plt.show()

        """""""""""""""""""""""""""
        get EXT values for each word
        """""""""""""""""""""""""""
        if dist_type_str == 'LD':
            EXT_values_LD = list(G.degree(nodes))
        if dist_type_str == 'FDL':
            EXT_values_FDL = list(G.degree(nodes))
        if dist_type_str == 'FDK':
            EXT_values_FDK = list(G.degree(nodes))

    LD_lst = []
    FDL_lst = []
    FDK_lst = []

    for item_LD, item_FDL, item_FDK in zip(EXT_values_LD, EXT_values_FDL, EXT_values_FDK):
        LD_lst.append(item_LD[1])
        FDL_lst.append(item_FDL[1])
        FDK_lst.append(item_FDK[1])

    EXT_values_all = pd.DataFrame(
        {'LD': LD_lst,
         'FDL': FDL_lst,
         'FDM': FDK_lst
         }, index=LD.index)

    EXT_values_all.to_csv(save_to_csv + 'EXT_values_threshold.csv')

#   print edges and nodes
# print(G.nodes())    #   returns a list of nodes
# print(G.edges())    #   returns a list of edges

"""
Now we calculate INT-values which are based on edges which only exist if the dist between two words
is smaller or equal 0.25.
"""

if WEIGHTED:
    for dist_type, dist_type_str in zip([LD, FDL, FDK], ['LD', 'FDL', 'FDK']):
        G = nx.Graph()  # creates an empty graph
        #   add nodes
        nodes = dist_type.columns
        G.add_nodes_from(nodes)

        #   add edges
        edges = []
        for word_1 in nodes:
            for word_2 in nodes:
                if word_1 is not word_2:
                    #   if words are in the df multiple times, we choose the first occurrence
                    if dist_type.at[word_1, word_2].shape == (2,):
                        if dist_type.at[word_1, word_2][0] != 1:
                            G.add_edge(word_1, word_2, weight=1-dist_type.at[word_1, word_2])
                    else:
                        if dist_type.at[word_1, word_2] != 1:
                            G.add_edge(word_1, word_2, weight=1-dist_type.at[word_1, word_2])

        #   draw graph
        nx.draw(G)
        plt.show()

        """""""""""""""""""""""""""
        get EXT values for each word
        """""""""""""""""""""""""""
        if dist_type_str == 'LD':
            EXT_values_LD = list(G.degree(nodes, weight='weight'))
        if dist_type_str == 'FDL':
            EXT_values_FDL = list(G.degree(nodes, weight='weight'))
        if dist_type_str == 'FDK':
            EXT_values_FDK = list(G.degree(nodes, weight='weight'))

    LD_lst = []
    FDL_lst = []
    FDK_lst = []

    for item_LD, item_FDL, item_FDK in zip(EXT_values_LD, EXT_values_FDL, EXT_values_FDK):
        LD_lst.append(item_LD[1])
        FDL_lst.append(item_FDL[1])
        FDK_lst.append(item_FDK[1])

    EXT_values_all = pd.DataFrame(
        {'LD': LD_lst,
         'FDL': FDL_lst,
         'FDK': FDK_lst
         }, index=LD.index)

    EXT_values_all.to_csv(save_to_csv + 'EXT_values_weighted.csv')