"""""""""""""""""""""""""""""""""""""""""""""""""""
In this file we plot all effects we will use in the model.
"""""""""""""""""""""""""""""""""""""""""""""""""""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#   where to save outputs
save_to_png = './png_files/'

LongPhon = pd.read_csv('.\csv_outputs\LongPhon.csv', sep=",", header=int(), index_col=0)

LD = pd.read_csv('.\csv_outputs\LD_norm.csv', sep=",", header=int(), index_col=0)
IPA_lst = LD.columns.tolist()

words = pd.read_csv('.\csv_files\words_mixed_effects.csv', sep=",", index_col=0)
words['IPA_final'] = IPA_lst
words = words.set_index('IPA_final')
words.head()     # to display the first 5 lines of loaded data

prod = pd.read_csv('.\csv_files\observ_reduced.csv', index_col=0)
prod.columns = ['child_id', 'observ_id', 'age', 'production'] + IPA_lst
#   change "produces" to "1" and "NaN" to "0"
prod = prod.replace('produces', 1)
prod = prod.replace(np.nan, 0)
prod.head()


""" Length of words """
word_length_df = pd.DataFrame(LongPhon.length.value_counts(sort=False))

plt.grid(True)

plt.bar(word_length_df.index, word_length_df.length, zorder=1)
plt.xlabel('phoneme length of words')
plt.ylabel('number of words')
plt.title('Distribution of phoneme length of words')

plt.show()


""" Frequency of words """
LongPhon.hist(column='frequency', bins=range(0, 100000, 1000))
plt.grid(True)
plt.xlabel('frequency of words')
plt.ylabel('number of words')
plt.title('Distribution of frequency of words')
plt.show()


""" Age of Children """

age_df = pd.DataFrame(LongPhon.age.value_counts(sort=False))
plt.bar(age_df.index, age_df.age, zorder=1)
plt.grid(True)
plt.xlabel('age of child')
plt.ylabel('number')
plt.title('Distribution of age of children')

plt.show()


""" Frequency of the word's lexical class """
word_category_df = pd.DataFrame(LongPhon.category.value_counts(sort=False))

plt.grid(True)
plt.bar(word_category_df.index, word_category_df.category)
plt.xlabel('lexical class of words')
plt.ylabel('number of words')
plt.title('Distribution of lexical class of words')

plt.show()


""" Frequency of the word's ext_LD """
LongPhon.hist(column='ext_LD')
plt.grid(True)
plt.xlabel('EXT LD')
plt.ylabel('number of words')
plt.title('Distribution of EXT-values based on the LD network')

plt.show()


""" Frequency of the word's ext_FDL """
LongPhon.hist(column='ext_FDL')
plt.grid(True)
plt.xlabel('EXT FDL')
plt.ylabel('number of words')
plt.title('Distribution of EXT-values based on the FDL network')

plt.show()


""" Frequency of the word's ext_FDM """
LongPhon.hist(column='ext_FDM')
plt.grid(True)
plt.xlabel('EXT FDM')
plt.ylabel('number of words')
plt.title('Distribution of EXT-values based on the FDM network')

plt.show()


""" Frequency of the word's int_LD """
LongPhon.hist(column='int_LD')
plt.grid(True)
plt.xlabel('INT LD')
plt.ylabel('number of words')
plt.title('Distribution of INT-values based on the LD network')

plt.show()


""" Frequency of the word's int_FDL """
plt.grid(True)
LongPhon.hist(column='int_FDL')
plt.xlabel('INT FDL')
plt.ylabel('number of words')
plt.title('Distribution of INT-values based on the FDL network')

plt.show()


""" Frequency of the word's int_FDM """
LongPhon.hist(column='int_FDM')
plt.grid(True)
plt.xlabel('INT FDM')
plt.ylabel('number of words')
plt.title('Distribution of INT-values based on the FDM network')

plt.show()


""" AoA of words """
word_aoa_df = pd.DataFrame(words.AoA.value_counts(sort=False))

plt.grid(True)
plt.bar(word_aoa_df.index, word_aoa_df.AoA)
plt.xlabel('age of acquisition of words')
plt.ylabel('number of words')
plt.title('Distribution of age of acquisition of words')

plt.show()


""" Production size per child/observation """
children_prod_size_df = pd.DataFrame(prod['production'].value_counts(sort=False), columns=['count'])

plt.hist(prod['production'], bins=75)

plt.grid(True)

plt.xlabel('productive vocabulary size')
plt.ylabel('number of observations')
plt.title('Distribution of productive vocabulary size')

plt.show()

