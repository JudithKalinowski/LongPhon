"""""""""""""""""""""""""""""""""""""""""""""
In this file we bring all files together
and set up the model.
"""""""""""""""""""""""""""""""""""""""""""""

"""
1. load needed packages
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px                 # for the heatmaps
import seaborn as sns

#   where to save outputs
save_to_csv = './csv_files/'
save_to_png = './png_files/'


"""
2. Import needed files
"""

words = pd.read_csv('.\csv_files\words_mixed_effects.csv', sep=",")
words.head()     # to display the first 5 lines of loaded data

LD = pd.read_csv('.\csv_outputs\LD_norm.csv', sep=",", header=int(), index_col=0)
LD = LD.set_index(LD.columns)

FDL = pd.read_csv('.\csv_outputs\FDL_norm_eucl.csv', sep=",", header=int(), index_col=0)
FDL = FDL.set_index(FDL.columns)

FDK = pd.read_csv('.\csv_outputs\FDK_norm_eucl.csv', sep=",", header=int(), index_col=0)
FDK = FDK.set_index(FDK.columns)


"""
3. Analyse phonological distances
"""

IPA_lst = LD.columns.tolist()

counts, bins = np.histogram(FDK, bins=20)
plt.stairs(counts, bins, label='FDK')
#
counts, bins = np.histogram(FDL, bins=20)
plt.stairs(counts, bins, label='FDL')
#
counts, bins = np.histogram(LD, bins=20)
plt.stairs(counts, bins, label='LD')
#
plt.axvline(x=0.25, color='black', linestyle='dotted', label='Threshold for phonological connectedness')
plt.xlabel('phonological distance')
plt.ylabel('number of word pairs x 2')
plt.legend(loc='upper left')
# plt.title('Distribution of three different phonological distance measures')
plt.show()


def exampl_wordpair_dist(a, b, word_1, word_2):
    print('LD_dist of' + word_1 + ' and ' + word_2 + ': ', LD.iloc[a, b], "\n",
          'FDL_dist of' + word_1 + ' and ' + word_2 + ': ', FDL.iloc[a, b], "\n",
          'FDK_dist of' + word_1 + ' and ' + word_2 + ': ', FDK.iloc[a, b])


exampl_wordpair_dist(20, 120, IPA_lst[20], IPA_lst[120])
exampl_wordpair_dist(220, 320, IPA_lst[220], IPA_lst[320])
exampl_wordpair_dist(420, 520, IPA_lst[420], IPA_lst[520])
exampl_wordpair_dist(620, 650, IPA_lst[620], IPA_lst[650])


def heatmap_pd(distance_matrix, title, data_type='png',  m=0, n=699):
    """
    :param: distance_matrix: the distance matrix we want to plot
    :param: m, n is the interval you want to have a look at
    :param: title: title of the plot to save
    :param: data_type: data type of the saved image
    :return: a heatmap with the distances
    """
    fig = px.imshow(distance_matrix.iloc[m:n, m:n], zmin=0, zmax=1, color_continuous_scale='Blues_r', width=600, height=550)
    fig.show()
    if data_type == 'svg':
        fig.write_image(save_to_png + title + '_' + str(m) + '_' + str(n) + '.svg')
    else:
        fig.write_image(save_to_png + 'PD/' + title + '_' + str(m) + '_' + str(n) + '.png')


a = 75
b = 85

for dist_type, title in zip([LD, FDL, FDK], ['LD', 'FDL', 'FDK']):
    heatmap_pd(dist_type, title, 'svg', m=a, n=b)
    heatmap_pd(dist_type, title, m=a, n=b)


#   get correlation of distances

# To find the correlation among the
# columns of df1 and df2 along the column axis

# create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

LD_FDK = LD.corrwith(FDK, axis=0)
x2 = words.IPA_length_nlb
y2 = LD_FDK[:]
sns.regplot(x=x2, y=y2, order=1, label="corr(LD, FDK)", marker=".")
# plt.scatter(words.IPA_length_nlb, LD_FDK[:], label="corr(LD, FDK)", alpha=0.5, marker=".")

FDL_FDK = FDL.corrwith(FDK, axis=0)
x3 = words.IPA_length_nlb
y3 = FDL_FDK[:]
sns.regplot(x=x3, y=y3, order=1, label="corr(FDL, FDK)", marker=".")
# plt.scatter(words.IPA_length_nlb, FDL_FDM[:], label="corr(FDL, FDM)", alpha=0.5, marker=".")

LD_FDL = LD.corrwith(FDL, axis=0)
x1 = words.IPA_length_nlb
y1 = LD_FDL[:]
sns.regplot(x=x1, y=y1, order=1, label="corr(LD, FDL)", marker=".")
# plt.scatter(words.IPA_length_nlb, LD_FDL[:], label="corr(LD, FDL)", alpha=0.5, marker=".")


#   Plot differences between the distances in a heatmap
FDL_LD_sub = abs(np.subtract(FDL, LD))
FDL_FDK_sub = abs(np.subtract(FDL, FDK))
FDK_LD_sub = abs(np.subtract(FDK, LD))

heatmap_pd(FDL_LD_sub, title="FDL_LD_sub", m=a, n=b)
heatmap_pd(FDL_FDK_sub, title="FDL_FDK_sub", m=a, n=b)
heatmap_pd(FDK_LD_sub, title="FDK_LD_sub", m=a, n=b)
