"""""""""""""""""""""""""""""""""""""""""""""
in this file we create the final network
which the children will have at the end.
This network includes all words from the CDI.
"""""""""""""""""""""""""""""""""""""""""""""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

#   import needed files
LD = pd.read_csv('.\csv_files\LD_norm_eucl.csv', sep=",", header=int(), index_col=0)
FDK = pd.read_csv('.\csv_files\FDK_norm_eucl.csv', sep=",", header=int(), index_col=0)
FDL = pd.read_csv('.\csv_files\FDL_norm_eucl.csv', sep=",", header=int(), index_col=0)

save_to_png = './png_outputs/'

"""""""""""""""""""""""""""
create graph
"""""""""""""""""""""""""""


def final_network(data, dist_type):
    """
    :param: data - list with all words in their ipa transcription
    :param: dist_type - sting of dist type (LD, FDL or FDK)
    :return: the final network of the children
    """
    #   creates an empty graph
    g = nx.Graph()

    #   add nodes
    nodes = data.columns
    g.add_nodes_from(nodes)

    #   add edges
    edges = []
    for word_1 in nodes:
        for word_2 in nodes:
            if word_1 is not word_2:
                #   if words are in the df multiple times, we choose the first occurrence
                if data.at[word_1, word_2].shape == (2,):
                    if data.at[word_1, word_2][0] <= 0.25:
                        edges.append((word_1, word_2))
                else:
                    if data.at[word_1, word_2] <= 0.25:
                        edges.append((word_1, word_2))
    g.add_edges_from(edges)

    #   draw graph
    graph = nx.draw_spring(g, with_labels=False, node_size=9, font_size=8)
    plt.show()

    plt.savefig(save_to_png + 'final_network_%s.png' % dist_type, bbox_inches='tight', dpi=1000)
    # plt.close()
    return graph
