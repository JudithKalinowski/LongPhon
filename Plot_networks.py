"""""""""""""""""""""""""""""""""""""""""""""
In this file we bring all files together
and set up the model.
"""""""""""""""""""""""""""""""""""""""""""""

"""
load needed packages
"""

import pandas as pd
import numpy as np
from mycolorpy import colorlist as mcp
from Individual_networks import ind_networks
from Final_network import final_network


#   where to save outputs
save_to_csv = './csv_files/'
save_to_png = './png_files/'


"""
1. Get files in
"""

prod = pd.read_csv('.\csv_files\observ.csv')
#   change "produces" to "1" and "NaN" to "0"
prod = prod.replace('produces', 1)
prod = prod.replace(np.nan, 0)
prod.head()


"""
2. Get files with phonological distances
"""

#   import needed files
LD = pd.read_csv('.\csv_outputs\LD_norm.csv', sep=",", header=int(), index_col=0)
LD = LD.set_index(LD.columns)

FDL = pd.read_csv('.\csv_outputs\FDL_norm_eucl.csv', sep=",", header=int(), index_col=0)
FDL = FDL.set_index(FDL.columns)

FDK = pd.read_csv('.\csv_outputs\FDK_norm_eucl.csv', sep=",", header=int(), index_col=0)
FDK = FDK.set_index(FDK.columns)

"""
4. Get networks
"""

#   final networks

final_network(LD, 'LD')
final_network(FDL, 'FDL')
final_network(FDK, 'FDK')


#   individual networks

#   choose colors for graph
color1 = mcp.gen_color(cmap="Blues", n=4)
print(color1)

#   group dataframe by child id
child_data = prod.groupby('child_id')
firsts = child_data.first()  #   get the first row of all children
child_ids = firsts.index.tolist()

ind_networks(child_ids, LD, 'LD')
ind_networks(child_ids, FDL, 'FDL')
ind_networks(child_ids, FDK, 'FDK')