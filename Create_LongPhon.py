"""""""""""""""""""""""""""""""""""""""""""""""""""
In this file we create a csv which 
will be used in the logistic regression model
"""""""""""""""""""""""""""""""""""""""""""""""""""

import pandas as pd
import numpy as np
from sklearn import preprocessing

#   where to save outputs
save_to_csv = './csv_files/'

#   import files
LD = pd.read_csv('.\csv_outputs\LD_norm.csv', sep=",", header=int(), index_col=0)
FDL = pd.read_csv('.\csv_outputs\FDL_norm_eucl.csv', sep=",", header=int(), index_col=0)
FDK = pd.read_csv('.\csv_outputs\FDK_norm_eucl.csv', sep=",", header=int(), index_col=0)

IPA_lst = LD.columns.tolist()

prod = pd.read_csv('.\csv_files\observ_reduced.csv', index_col=0)
prod.columns = ['child_id', 'observ_id', 'age', 'production'] + IPA_lst
#   change "produces" to "1" and "NaN" to "0"
prod = prod.replace('produces', 1)
prod = prod.replace(np.nan, 0)
prod.head()

words = pd.read_csv('.\csv_files\words_mixed_effects.csv', sep=",", index_col=0)
words['IPA_final'] = IPA_lst
words = words.set_index('IPA_final')
words.head()     # to display the first 5 lines of loaded data

INT_values_LD = pd.read_csv('.\csv_files\INT_values_LD_df.csv', sep=",", header=int(), index_col=0)
INT_values_FDL = pd.read_csv('.\csv_files\INT_values_FDL_df.csv', sep=",", header=int(), index_col=0)
INT_values_FDK = pd.read_csv('.\csv_files\INT_values_FDK_df.csv', sep=",", header=int(), index_col=0)

EXT_values = pd.read_csv('.\csv_files\EXT_values.csv', sep=",", header=int(), index_col=0)

LongPhon_INT = pd.DataFrame(columns=['child', 'word', 'age', 'produced', 'int_LD', 'int_FDL', 'int_FDM'])

for row in range(1, len(prod.index)):
    if prod.at[row - 1, 'child_id'] == prod.at[row, 'child_id']:
        for col in range(4, len(prod.columns)):
            if prod.iat[row - 1, col] == 0:
                LongPhon_INT.loc[len(LongPhon_INT.index)] = [prod.at[row, 'child_id'], prod.columns.tolist()[col],
                                                             prod.at[row, 'age'], prod.iat[row, col],
                                                             INT_values_LD.iat[row - 1, col - 4],
                                                             INT_values_FDL.iat[row - 1, col - 4],
                                                             INT_values_FDK.iat[row - 1, col - 4]]


LongPhon_INT['ext_LD'], LongPhon_INT['ext_FDL'], LongPhon_INT['ext_FDK'] = [np.nan, np.nan, np.nan]

for row in range(0, len(LongPhon_INT)):
    target_word = LongPhon_INT.at[row, 'word']
    ext_LD = EXT_values.at[target_word, 'LD']
    ext_FDL = EXT_values.at[target_word, 'FDL']
    ext_FDK = EXT_values.at[target_word, 'FDL']
    ##
    LongPhon_INT.at[row, 'ext_LD'] = ext_LD
    LongPhon_INT.at[row, 'ext_FDL'] = ext_FDL
    LongPhon_INT.at[row, 'ext_FDK'] = ext_FDK

LongPhon_INT['length'], LongPhon_INT['frequency'], LongPhon_INT['category'] = [np.nan, np.nan, np.nan]

for row in range(0, len(LongPhon_INT)):
    target_word = LongPhon_INT.at[row, 'word']
    word_length = words.at[target_word, 'IPA_length_final']
    word_frequ = words.at[target_word, 'Frequency']
    word_cat = words.at[target_word, 'lexical_class']
    ##
    LongPhon_INT.at[row, 'length'] = word_length
    LongPhon_INT.at[row, 'frequency'] = word_frequ
    LongPhon_INT.at[row, 'category'] = word_cat

LongPhon_INT.to_csv(save_to_csv + 'LongPhon.csv')


"""
We also need to normalise the INT- and EXT-values. For the EXT-values we can do this
over the whole data frame. However, as the INT-values grow with a growing network,
we must normalise the INT-values within each child and each observation. Otherwise
the model is confused, i.e. big values in an early observation are comparatively small
in later observations.
"""

# 1. group by child
# 2. group by age
# 3. normalise over all INT-values in the child/age values

LongPhon = LongPhon_INT.copy()
to_normalise = LongPhon_INT.groupby(['child', 'age'])

INT_LD_norm_lst = []
INT_FDL_norm_lst = []
INT_FDK_norm_lst = []

min_max_scaler = preprocessing.MinMaxScaler()

for child in to_normalise:
    INT_LD_norm = child[1].int_LD.values.reshape(-1, 1)  # returns a numpy array
    INT_LD_norm_scaled = min_max_scaler.fit_transform(INT_LD_norm)
    INT_LD_norm_lst.append(INT_LD_norm_scaled)
    ##
    INT_FDL_norm = child[1].int_FDL.values.reshape(-1, 1)  # returns a numpy array
    INT_FDL_norm_scaled = min_max_scaler.fit_transform(INT_FDL_norm)
    INT_FDL_norm_lst.append(INT_FDL_norm_scaled)
    ##
    INT_FDK_norm = child[1].int_FDK.values.reshape(-1, 1)  # returns a numpy array
    INT_FDK_norm_scaled = min_max_scaler.fit_transform(INT_FDK_norm)
    INT_FDK_norm_lst.append(INT_FDK_norm_scaled)

flatten_LD = [item for sublist in INT_LD_norm_lst for item in sublist]
LongPhon['int_LD'] = [item[0] for item in flatten_LD]
flatten_FDL = [item for sublist in INT_FDL_norm_lst for item in sublist]
LongPhon['int_FDL'] = [item[0] for item in flatten_FDL]
flatten_FDK = [item for sublist in INT_FDK_norm_lst for item in sublist]
LongPhon['int_FDK'] = [item[0] for item in flatten_FDK]

#   Normalise EXT values (no grouping necessary; see explanation above)
EXT_LD_norm_lst = []
EXT_FDL_norm_lst = []
EXT_FDK_norm_lst = []

EXT_LD_norm = LongPhon.ext_LD.values.reshape(-1, 1)  # returns a numpy array
EXT_LD_norm_scaled = min_max_scaler.fit_transform(EXT_LD_norm)
EXT_LD_norm_lst.append(EXT_LD_norm_scaled)
##
EXT_FDL_norm = LongPhon.ext_FDL.values.reshape(-1, 1)  # returns a numpy array
EXT_FDL_norm_scaled = min_max_scaler.fit_transform(EXT_FDL_norm)
EXT_FDL_norm_lst.append(EXT_FDL_norm_scaled)
##
EXT_FDK_norm = LongPhon.ext_FDM.values.reshape(-1, 1)  # returns a numpy array
EXT_FDK_norm_scaled = min_max_scaler.fit_transform(EXT_FDK_norm)
EXT_FDK_norm_lst.append(EXT_FDK_norm_scaled)

flatten_LD_ext = [item for sublist in EXT_LD_norm_lst for item in sublist]
LongPhon['ext_LD'] = [item[0] for item in flatten_LD_ext]
flatten_FDL_ext = [item for sublist in EXT_FDL_norm_lst for item in sublist]
LongPhon['ext_FDL'] = [item[0] for item in flatten_FDL_ext]
flatten_FDK_ext = [item for sublist in EXT_FDK_norm_lst for item in sublist]
LongPhon['ext_FDM'] = [item[0] for item in flatten_FDK_ext]


LongPhon.to_csv(save_to_csv + 'LongPhon.csv')


"""
Check for collinearity
"""

LongPhon = pd.read_csv('.\csv_outputs\LongPhon.csv', sep=",", header=int(), index_col=0)

print(LongPhon['int_LD'].corr(LongPhon['ext_LD']),
      LongPhon['int_FDL'].corr(LongPhon['ext_FDL']),
      LongPhon['int_FDM'].corr(LongPhon['ext_FDM']))
