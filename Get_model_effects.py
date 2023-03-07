"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
in this file we calculate the fixed effects
"word_length", "word_frequency", "age_of_acquisition_averaged",
"age_or_acquisition_individual", "observation_gap" and plot them.
We also plot the INT- and EXT-values.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#   where to save outputs
save_to_csv = './csv_files/'
save_to_png = './png_files/'

"""
1. Get files in
"""

part = pd.read_csv('.\csv_files\participants.csv', sep=";")
part.head()     # to display the first 5 lines of loaded data

words = pd.read_csv('.\csv_files\words.csv', sep=";")
words.head()     # to display the first 5 lines of loaded data

prod = pd.read_csv('.\csv_files\observ.csv', sep=",")
prod.head()     # to display the first 5 lines of loaded data

#   import files
LD = pd.read_csv('.\csv_files\LD_norm.csv', sep=",", header=int(), index_col=0)

IPA_lst = LD.columns.tolist()


""" Frequency file """
f = open('.\csv_files\words.freq', "r", encoding='UTF-8')

freq = []
word = []
line = f.readline()
while line:
    data = line.split("\t")[0]
    freq.append(int(data[:8]))
    word.append(data[8:].strip())

    line = f.readline()

frequ_df = pd.DataFrame({"freq": freq, "word": word})
frequ_df = frequ_df.set_index('word')
print(frequ_df.head)


""" Production File """
#   change "produces" to "1" and "NaN" to "0"
prod = prod.replace('produces', 1)
prod = prod.replace(np.nan, 0)

#   we dropped some words which are not useful for our analysis, so we also have to drop them here
num_item_id = []
for word_id in words.num_item_id.tolist():
    num_item_id.append(str(word_id))
final_columns = num_item_id + ['original_id', 'data_id', 'age', 'production']
prod_reduced = prod.drop(columns=[col for col in prod if col not in final_columns])
prod_reduced = prod_reduced.set_axis([['child_id', 'observ_id', 'age', 'production'] + IPA_lst],
                                     axis=1)

prod_reduced.head()     # to display the first 5 lines of loaded data


"""
2. Calculate word length
"""

''' delete elements in IPA final transcriptions of words '''
for idx, ele in enumerate(IPA_lst):
    IPA_lst[idx] = ele.replace('ˈ', '')
for idx, ele in enumerate(IPA_lst):
    IPA_lst[idx] = ele.replace('ˌ', '')
for idx, ele in enumerate(IPA_lst):
    IPA_lst[idx] = ele.replace(' ', '')
for idx, ele in enumerate(IPA_lst):
    IPA_lst[idx] = ele.replace('ː', '')
for idx, ele in enumerate(IPA_lst):
    IPA_lst[idx] = ele.replace('.', '')
for idx, ele in enumerate(IPA_lst):
    IPA_lst[idx] = ele.replace('"', '')
for idx, ele in enumerate(IPA_lst):
    IPA_lst[idx] = ele.replace('1', '')
for idx, ele in enumerate(IPA_lst):
    IPA_lst[idx] = ele.replace('2', '')
for idx, ele in enumerate(IPA_lst):
    IPA_lst[idx] = ele.replace('3', '')


def get_word_length(ipa_source):
    """
    :param ipa_source: choose from which source we want to use the IPA transcriptions (IPA_espeak, IPA_naob)
    :return: list of the length of each individual word; ordered in the same order as the "words" df
    """
    word_length_lst = []
    for w in ipa_source:
        if type(w) == float:
            word_length_lst.append(None)
        else:
            word_length_lst.append(len(w))
    return word_length_lst

word_length_nlb = get_word_length(words.IPA_nlb)
word_length_ipa_lst_final = get_word_length(IPA_lst)

words['IPA_length_nlb'] = word_length_nlb
words['IPA_length_final'] = word_length_ipa_lst_final


"""
3. Get frequency of words (from Norwegian Web As Corpus)
"""

frequ_word_lst = frequ_df.index.to_list()
word_lst_wordbank = words.definition_new.tolist()

frequ_lst = []
for word in word_lst_wordbank:
    if word in frequ_word_lst:
        if np.ndim(frequ_df.loc[word]['freq']) == 1:
            frequ_lst.append(frequ_df.loc[word]['freq'][0])
        elif np.ndim(frequ_df.loc[word]['freq']) == 0:
            frequ_lst.append(frequ_df.loc[word]['freq'])
    else:
        frequ_lst.append(None)

words['Frequency'] = frequ_lst


"""
4. Get AoA (also from wordbank/ By-word-summary data); averaged over all children
Edit: Not used in the end.
"""

word_perc = pd.read_csv("./csv_files/word_acqu_percentages.csv", header=int(), index_col=0)

""" we deleted some words from the item data because they are not useful for our research
Therefore, we also must delete these words in the item_data dataframe """

item_id_lst = words.item_id.tolist()
reduced_word_perc = word_perc[word_perc.index.isin(item_id_lst)]

mean_aoa_df = pd.DataFrame(columns=['word_id', 'word', 'mean_aoa'])
words_lst = words.definition_new.tolist()

n = 0
for row in reduced_word_perc.index:
    row_lst = reduced_word_perc.loc[row, :].values.tolist()
    new_lst = []
    for item in row_lst[2:]:
        if item < 0.5:
            new_lst.append(item)
    mean_aoa_df.loc[len(mean_aoa_df)] = [row, words_lst[n], len(new_lst) + 16]
    n = n+1

mean_aoa = mean_aoa_df.set_index('word_id')
mean_aoa.to_csv(save_to_csv + 'aoa_df.csv')

words["AoA"] = mean_aoa.mean_aoa.tolist()
words.to_csv(save_to_csv + 'words_mixed_effects.csv')


"""
5. Get observation gaps
Edit: Not used in the end.
"""

gaps_lst = []

for i in range(0, len(prod_reduced)):
    if i == 0:
        diff = None
        gaps_lst.append(diff)
    else:
        child = prod_reduced.iloc[i]['child_id']
        child_obs_before = prod_reduced.iloc[i - 1]['child_id']
        if child == child_obs_before:
            diff = prod_reduced.iloc[i]['age'] - prod_reduced.iloc[i-1]['age']
            gaps_lst.append(diff)
        else:
            #   set difference = None for initial observation for each child
            diff = None
            gaps_lst.append(diff)

prod_reduced.insert(4, 'obs_gap', gaps_lst)
prod_reduced.to_csv(save_to_csv + 'observ_reduced.csv')


"""
6. INT and EXT values
"""

EXT_values = pd.read_csv("./csv_files/EXT_values.csv", header=int(), index_col=0)
INT_values_LD = pd.read_csv("./csv_files/INT_values_LD_df.csv", header=int(), index_col=0)
INT_values_FDL = pd.read_csv("./csv_files/INT_values_FDL_df.csv", header=int(), index_col=0)
INT_values_FDK = pd.read_csv("./csv_files/INT_values_FDK_df.csv", header=int(), index_col=0)

"""
7. Age of first production of a word by child
Edit: Not used in the end.
"""

child_id_lst = list(dict.fromkeys(prod.original_id.tolist()))
AoP_df = pd.DataFrame(columns=LD.columns, index=child_id_lst)
##
for row in range(len(prod_reduced.index)):
    for word in LD.columns:
        if prod_reduced.iloc[row][word] == 1:
            if row == 0:
                a = prod_reduced.iloc[row]['child_id']
                AoP_df.loc[int(a)][word] = prod_reduced.iloc[row]['age']
                print(prod_reduced.iloc[row]['age'])
            else:
                if prod_reduced.iloc[row]['child_id'] == prod_reduced.iloc[row - 1]['child_id']:
                    if prod_reduced.iloc[row-1][word] == 0:
                        a = prod_reduced.iloc[row]['child_id']
                        AoP_df.loc[int(a)][word] = prod_reduced.iloc[row]['age']
                        print(prod_reduced.iloc[row]['age'])
                else:
                    a = prod_reduced.iloc[row]['child_id']
                    AoP_df.loc[int(a)][word] = prod_reduced.iloc[row]['age']
                    print(prod_reduced.iloc[row]['age'])

AoP_df.to_csv(save_to_csv + 'AoP_df.csv')


"""
8. Plot distribution of fixed effects
"""

""" Gap between two observations of the same child """

obs_gaps_df = pd.DataFrame(prod_reduced['obs_gap'].value_counts(sort=False), columns=['count'])

plt.bar(obs_gaps_df.index.get_level_values(0), obs_gaps_df['count'])
plt.xlabel('gap size between two observations')
plt.ylabel('number of gaps')
plt.title('Distribution of gaps between two observations')
plt.axis([0, 16, 0, 1350])
plt.grid(True)
plt.show()


""" Length of words """
word_length_df = pd.DataFrame(words.IPA_length_nlb.value_counts(sort=False))

plt.grid(True)

plt.bar(word_length_df.index, word_length_df.IPA_length_nlb, zorder=1)
plt.xlabel('phoneme length of words')
plt.ylabel('number of words')
plt.title('Distribution of phoneme length of words')
plt.axis([0, 20, 0, 160])
plt.xticks(range(20))

plt.show()


""" Frequency of words """
plt.grid(True)

plt.hist(words.Frequency, bins=range(0, 10000, 100))
plt.xlabel('frequency of words')
plt.ylabel('number of words')
plt.title('Distribution of frequency of words')
plt.show()


""" Age of Children """
children_age_df = pd.DataFrame(prod_reduced['age'].value_counts(sort=False), columns=['count'])

plt.bar(children_age_df.index.get_level_values(0), children_age_df['count'])

plt.grid(True)

plt.xlabel('age of child')
plt.ylabel('number of observations')
plt.title('Distribution of age of children')
plt.axis([15, 38, 0, 700])

plt.show()


""" Frequency of words """
word_category_df = pd.DataFrame(words.lexical_category.value_counts(sort=False))

plt.grid(True)
plt.bar(word_category_df.index, word_category_df.lexical_category)
plt.xlabel('lexical category of words')
plt.ylabel('number of words')
plt.title('Distribution of lexical category of words')

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
children_prod_size_df = pd.DataFrame(prod_reduced['production'].value_counts(sort=False), columns=['count'])

plt.hist(prod_reduced['production'], bins=75)

plt.grid(True)

plt.xlabel('productive vocabulary size')
plt.ylabel('number of observations')
plt.title('Distribution of productive vocabulary size')
# plt.axis([0, 750, 0, 750])

plt.show()
