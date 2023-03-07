"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
in this file we take the SAMPA transcription from NLB
and convert it to IPA.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import pandas as pd
from Convert_sampa_ipa import sampa_to_ipa

#   where to save outputs
save_to_csv = './csv_files/'
save_to_png = './png_files/'

words = pd.read_csv('./csv_files/words.csv', sep=";", header=int())

NLB = pd.read_csv('C:/Users/judit/OneDrive/Desktop/NLB.csv', sep=";", encoding="mbcs", header=None)
cols = ['word', 'sampa', 'eigenschaften', 'garb', 'ortlang', 'pronlang', 'code',
        'pronvar', 'decomp', 'spell', 'freq', 'update', 'orig', 'lemma', 'id', 'status']
NLB.columns = cols

NLB_sampa_lst = NLB.sampa.tolist()
for idx, ele in enumerate(NLB_sampa_lst):
    NLB_sampa_lst[idx] = ele.replace(' ', '')

sampa_ipa_df = pd.DataFrame({'sampa': NLB_sampa_lst, 'ipa': None})
sampa_ipa_df = sampa_ipa_df.set_index('sampa')

for element in NLB_sampa_lst:
    ipa = sampa_to_ipa(element)
    print(ipa)
    sampa_ipa_df.loc[element, 'ipa']