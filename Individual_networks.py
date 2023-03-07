"""""""""""""""""""""""""""""""""""""""""""""
in this file we create the monthly networks
of each child.
This network includes all words the individual child
knows at each time point and includes all words the 
child had acquired within the months before.
"""""""""""""""""""""""""""""""""""""""""""""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp

save_to_png = './png_files/'

#   import needed files

#   LD (levenshtein distance), FD (euclidean feature distance), FDS (euclidean feature distance with syllable permut.)
LD = pd.read_csv('.\csv_outputs\LD_norm_eucl.csv', sep=",", header=int(), index_col=0)
FDL = pd.read_csv('.\csv_outputs\FDL_norm_eucl.csv', sep=",", header=int(), index_col=0)
FDK = pd.read_csv('.\csv_outputs\FDK_norm_eucl.csv', sep=",", header=int(), index_col=0)

#   set df index == df cols
LD = LD.set_axis(LD.columns.tolist(), axis='index')
FDL = FDL.set_axis(FDL.columns.tolist(), axis='index')
FDK = FDK.set_axis(FDK.columns.tolist(), axis='index')

IPA_lst = LD.columns.tolist()

prod = pd.read_csv('.\csv_files\observ_reduced.csv', index_col=0)
prod = prod.set_axis(['child_id', 'observ_id', 'age', 'production'] + IPA_lst, axis=1)
prod.head()     # to display the first 5 lines of loaded data

"""""""""""""""""""""""""""
create graph
"""""""""""""""""""""""""""
#   choose colors for graph
color1 = mcp.gen_color(cmap="cool", n=4)

#   group dataframe by child id
child_data = prod.groupby('child_id')
firsts = child_data.first()  #   get the first row of all children
child_ids = firsts.index.tolist()


def ind_networks(data, dist_df, dist_type):
    plt.close()
    for id in data[:4]:
        #   Finding the values contained in an individual child
        #   group by child id
        child = child_data.get_group(id)
        #   change index of individual child df to 1 to n
        child.index = pd.RangeIndex(start=0, stop=len(child.index), step=1)
        for r in range(len(child.index)):
            obs = child.loc[[r], 'ˈɛʉ':]
            for word in obs.columns:
                if obs.at[r, word] == 0:
                    obs = obs.drop(word, axis=1)
            #   create an empty graph
            G = nx.Graph()
            #   add nodes
            #   the nodes are now all words which had a 1 (=can be produced) in the CDI for this child
            nodes = obs.columns.tolist()
            G.add_nodes_from(nodes)

            #   add edges
            edges = []
            for word_1 in nodes:
                for word_2 in nodes:
                    if word_1 is not word_2:
                        if dist_df.at[word_1, word_2] <= 0.25:
                            edges.append((word_1, word_2))
            G.add_edges_from(edges)

            nx.draw(G, with_labels=True, node_size=9, font_size=8)
            plt.show()
            #   Saving the plot as an image
            plt.savefig(save_to_png + '%s_child_%d_obs_%d.png' % (dist_type, id, r), bbox_inches='tight', dpi=1000)
            plt.close()

