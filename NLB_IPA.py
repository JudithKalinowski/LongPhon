"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
in this file we take the words from wordbank, take the
SAMPA transcription from NLB, convert it to IPA, and add
it to the words df.
To run the file, you need to download the NLB-file because
it is too big to put on GitHub.
Link: https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-52/
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#   import needed packages
import numpy as np
import pandas as pd
from Convert_sampa_ipa import sampa_to_ipa

#   where to save outputs
save_to_csv = './csv_files/'
save_to_png = './png_files/'

#   import words df and delete items we do not want to include in our analysis
words = pd.read_csv('./csv_files/words.csv', sep=",", header=int())
deleted_items = pd.read_csv('./csv_files/deleted_items.csv', sep=";", names=['word'])
deleted_items_lst = deleted_items.word.tolist()
words = words[words['definition_new'].isin(deleted_items_lst) == False]
words = words.set_index('item_id')

word_lst = words.definition_new.tolist()
for idx, ele in enumerate(word_lst):
    if ele[len(ele)-1] == ' ':
        word_lst[idx] = ele[: -1]

words.pop('definition_new')
words.insert(loc=1, column='definition_new', value=word_lst)

#   import the NLB lexicon and set col names
NLB = pd.read_csv('./csv_files/NLB.csv', sep=",", encoding="mbcs", header=None)
cols = ['word', 'sampa', 'eigenschaften', 'garb', 'ortlang', 'pronlang', 'code',
        'pronvar', 'decomp', 'spell', 'freq', 'update', 'orig', 'lemma', 'id', 'status']
NLB.columns = cols
NLB = NLB.set_index('word')

#   check which words from our wordbank df are not in the NLB lexicon
lst = []
for word in words.definition_new.tolist():
    if word not in NLB.index.tolist():
        lst.append(word)
print(len(lst), 'of our wordbank items are not part of the NLB dictionary.\nThose words are:', lst)

#   get the ipa transcriptions from NLB of the words in our wordbank df
new_lst = []
wordbank_lst = words.definition_new.tolist()
for wb_item in wordbank_lst:
    if wb_item not in lst:
        if np.ndim(NLB.loc[wb_item]['sampa']) == 1:
            sampa = NLB.loc[wb_item]['sampa'][0]
            sampa_item = sampa.replace(' ', '')
            print(sampa_item)
        elif np.ndim(NLB.loc[wb_item]['sampa']) == 0:
            sampa = NLB.loc[wb_item]['sampa']
            sampa_item = sampa.replace(' ', '')
            print(sampa_item)
        ipa = sampa_to_ipa(sampa_item)
        print(ipa)
        new_lst.append(ipa)
    else:
        new_lst.append(None)
print(new_lst)

#   add the NLB ipa transcriptions to our words df
words['IPA_nlb'] = new_lst

#   save new words df
words.to_csv(save_to_csv + 'words_NLB.csv')
