"""""""""""""""""""""""""""""""""""""""""""""
In this file we calculate the children's
networks' INT values.

INT is an internally driven network growth scenario.
In the past it has also been referred to as PAT = Preferential Attachment.
- 'rich-get-richter' model of acquisition
- new nodes connect to node with highest degree
- prediction: the early vocabulary will consist
    of many similar-sounding forms
- PAT values: predicts a word will be learned
    if it connects to words in the existing
    network with higher mean degrees
- 2 assumptions of PAT-like growth:
    1. Low mean path length
    2. High clustering coefficient
- the value of a new node is the average
    degree of the known nodes it would attach to
"""""""""""""""""""""""""""""""""""""""""""""

import pandas as pd
import networkx as nx
import statistics
import matplotlib.pyplot as plt

#   import needed files

#   LD (levenshtein distance), FDL (euclidean feature distance Laing), FDS (euclidean feature distance Monaghan)
LD = pd.read_csv('.\csv_files\LD_norm.csv', sep=",", header=int(), index_col=0)
FDL = pd.read_csv('.\csv_files\FDL_norm_eucl.csv', sep=",", header=int(), index_col=0)
FDK = pd.read_csv('.\csv_files\FDK_norm_eucl.csv', sep=",", header=int(), index_col=0)

IPA_lst = LD.columns.tolist()

prod = pd.read_csv('.\csv_files\observ_reduced.csv', index_col=0)
prod = prod.set_axis(['child_id', 'observ_id', 'age', 'production'] + IPA_lst, axis=1)
prod.head()     # to display the first 5 lines of loaded data


"""""""""""""""""""""""""""
create network
"""""""""""""""""""""""""""

INT_values_LD_df = pd.DataFrame(columns=LD.columns, index=prod.index)
INT_values_FDL_df = pd.DataFrame(columns=LD.columns, index=prod.index)
INT_values_FDK_df = pd.DataFrame(columns=LD.columns, index=prod.index)

#   group dataframe by child id
child_data = prod.groupby('child_id')
firsts = child_data.first()  #   get the first row of all children
child_ids = firsts.index.tolist()


THRESHOLD = False
WEIGHTED = True


"""
Now we calculate INT-values which are based on edges which only exist if the dist between two words
is smaller or equal 0.25.
"""

if THRESHOLD:
    dist_count = 0

    for dist in [LD, FDL, FDK]:
        dist_count += 1
        row_count = 0
        for child_id in child_ids:
            #   Finding the values contained in an individual child
            child = child_data.get_group(child_id)    #   group by child id
            child.index = pd.RangeIndex(start=0, stop=len(child.index), step=1)     #   change index of individual child df to 1 to n
            for r in range(len(child.index)):
                obs = child.loc[[r], 'ˈɛʉ':]     #   get single observations for a single child with child_id "id"
                not_yet_known = []
                for word in obs.columns:
                    if obs.at[r, word] == 0:
                        not_yet_known.append(word)
                        obs = obs.drop(word, axis=1)
                #   create an empty graph
                G = nx.Graph()
                #   add nodes
                #   the nodes are now all words which had a 1 (=can be produced) in the CDI for this child
                nodes = obs.columns.tolist()
                G.add_nodes_from(nodes)

                dist.set_index(dist.columns, inplace=True)

                #   add edges
                edges = []
                for word_1 in nodes:
                    for word_2 in nodes:
                        if word_1 is not word_2:
                            #   not weighted edges, but with a threshold of the distance being smaller than 0.25
                            if dist.at[word_1, word_2] <= 0.25:
                                edges.append((word_1, word_2))
                G.add_edges_from(edges)

                """""""""""""""""""""""""""
                get INT-values
                """""""""""""""""""""""""""

                #   create list with yet-to-be-learned nodes and the mean degree of the nodes it would connect to
                INT_values = []
                #   loop through all yet-to-be-learned nodes
                for node in not_yet_known:
                    #   get list of the degrees of all connected nodes
                    connected_nodes_degree = []

                    G.add_node(node)    #   add not-yet-known node
                    node_edges = []     #   create list to save the edges the node would connect to
                    connected_nodes = []    #   create list to save the nodes the node would connec to
                    for word in nodes:  #   loop through all nodes in the existing network in this observation
                        if dist.at[word, node] <= 0.25:
                            node_edges.append((word, node))
                            connected_nodes.append(word)    #   add the connected node to the list of connected nodes
                    G.add_edges_from(node_edges)            #   add edges to the graph
                    for conn_node in connected_nodes:       #   loop though all connected nodes
                        conn_node_degree = len(G.edges(conn_node))      #   get a list of all edges of a connected node and its length
                        connected_nodes_degree.append(conn_node_degree)     #   add the degree of the connected node to a list of all degrees
                    connected_nodes_degree.sort()                  #   sort list so that we can get the median
                    if len(connected_nodes_degree) == 0:
                        median_degree = 0
                    else:
                        median_degree = statistics.median(connected_nodes_degree)
                    if dist_count == 1:
                        INT_values_LD_df.at[row_count, node] = median_degree
                    elif dist_count == 2:
                        INT_values_FDL_df.at[row_count, node] = median_degree
                    else:
                        INT_values_FDK_df.at[row_count, node] = median_degree
                    #mean_degree = np.mean(connected_nodes_degree)           #   get mean of all connected nodes' degrees
                    # PAT_values.append((node, mean_degree))           # store mean degree of connected nodes in list
                    INT_values.append((node, median_degree))  # store median degree of connected nodes in list
                print("INT-values for child %d, observation %d:" % (child_id, r), INT_values)
                row_count += 1
        if dist_count == 1:
            INT_values_LD_df.to_csv('./csv_files/' + 'INT_values_LD_df_threshold.csv')
        elif dist_count == 2:
            INT_values_FDL_df.to_csv('./csv_files/' + 'INT_values_FDL_df_threshold.csv')
        else:
            INT_values_FDK_df.to_csv('./csv_files/' + 'INT_values_FDK_df_threshold.csv')


"""
Now we calculate INT-values which are based on weighted edges in the network
"""

if WEIGHTED:
    dist_count = 0

    for dist in [LD, FDL, FDK]:
        dist_count += 1
        row_count = 0
        for child_id in child_ids:
            #   Finding the values contained in an individual child
            child = child_data.get_group(child_id)  # group by child id
            child.index = pd.RangeIndex(start=0, stop=len(child.index),
                                        step=1)  # change index of individual child df to 1 to n
            for r in range(len(child.index)):
                obs = child.loc[[r], 'ˈɛʉ':]  # get single observations for a single child with child_id "id"
                not_yet_known = []
                for word in obs.columns:
                    if obs.at[r, word] == 0:
                        not_yet_known.append(word)
                        obs = obs.drop(word, axis=1)
                #   create an empty graph
                G = nx.Graph()
                #   add nodes
                #   the nodes are now all words which had a 1 (=can be produced) in the CDI for this child
                nodes = obs.columns.tolist()
                G.add_nodes_from(nodes)

                dist.set_index(dist.columns, inplace=True)

                #   add edges
                edges = []
                for word_1 in nodes:
                    for word_2 in nodes:
                        if word_1 is not word_2:
                            #   weighted edges
                            if dist.at[word_1, word_2] != 1:
                                G.add_edge(word_1, word_2, weight=1-dist.at[word_1, word_2])

                """ get INT-values """
                #   create list with yet-to-be-learned nodes and the mean degree of the nodes it would connect to
                INT_values = []
                #   loop through all yet-to-be-learned nodes
                for node in not_yet_known:
                    W = G.copy(as_view=False)
                    W.add_node(node)  # add not-yet-known node
                    for word in nodes:  # loop through all nodes in the existing network in this observation
                        #   weighted edges
                        if dist.at[word, node] != 1:
                            W.add_edge(word, node, weight=1-dist.at[word, node])
                    node_neighbors = list(W.neighbors(node))
                    neighbors_degree = []
                    neighbors_degree_weighted = []
                    for neighbor in node_neighbors:
                        neighbors_degree_weighted.append(W.degree(neighbor, weight='weight'))
                        neighbors_degree.append(W.degree(neighbor))

                    connected_nodes_values = []
                    for i in range(0, len(neighbors_degree_weighted)):
                        if neighbors_degree_weighted[i] == 0:
                            connected_nodes_values.append(0)
                        else:
                            value_connected_node = neighbors_degree_weighted[i]/neighbors_degree[i]
                            connected_nodes_values.append(value_connected_node)

                    INT_value_node = sum(connected_nodes_values)

                    if dist_count == 1:
                        INT_values_LD_df.at[row_count, node] = INT_value_node
                    elif dist_count == 2:
                        INT_values_FDL_df.at[row_count, node] = INT_value_node
                    else:
                        INT_values_FDK_df.at[row_count, node] = INT_value_node

                    INT_values.append((node, INT_value_node))  # store INT values of connected nodes in list
                print("INT-values for child %d, observation %d:" % (child_id, r), INT_values)
                row_count += 1
        if dist_count == 1:
            INT_values_LD_df.to_csv('./csv_files/' + 'INT_values_LD_df_weighted.csv')
        elif dist_count == 2:
            INT_values_FDL_df.to_csv('./csv_files/' + 'INT_values_FDL_df_weighted.csv')
        else:
            INT_values_FDK_df.to_csv('./csv_files/' + 'INT_values_FDK_df_weighted.csv')
