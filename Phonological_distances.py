"""""""""""""""""""""""""""""""""""""""""""""""""""
In this file, we will calculate three different phonological distances of word pairs.
We will do this first by calculating the Levenshtein distance (LD) as a rough approximation.
We will then use Laing's (2022) phonological distance which aligns
vowels of the syllables, but keeps the syllable order (FDL).
Finally, we will also use syllable permutations (FDK).

Part one of this file is the definition of needed functions.
Part two is the execution of those functions.
"""""""""""""""""""""""""""""""""""""""""""""""""""

# packages to import

import numpy as np
from numpy.linalg import norm  # for euclidean distance
import pandas as pd
import itertools
from itertools import permutations
import plotly.express as px  # for the heatmaps
from sklearn.preprocessing import MinMaxScaler  # for normalisation
from scipy.spatial import distance

#   where to save outputs
save_to_csv = './csv_files/'
save_to_png = './png_files/'
save_to_svg = './png_outputs/'


"""""""""""""""""""""
PART 1:
DEFINE FUNCTIONS
"""""""""""""""""""""


def sum_numbers(numbers):
    total = 0
    for number in numbers:
        total += number
    return total


def contains_letter(string, letter):
    return letter in string


def calculate_features(target_word):
    """
    gets the phoneme features for a word
    :param: target_word: word in its IPA transcription
    :return: list of phoneme features for the target word
    """
    #   get features for each phoneme
    feat = []
    for p_1, p_2 in zip(target_word, target_word[1:]):
        if p_1 not in ignore_phon_lst:
            if p_2 not in ignore_phon_lst:
                #   check if a colon follows a phoneme; if so, put them together as one phoneme
                if not contains_letter(p_1, 'ː'):
                    if p_2 == 'ː':
                        p1 = "".join((p_1, p_2))
                        features = phon_feat.loc[[p1], 'advanced_tongue_root':].values
                        feat.append(features)
                    else:
                        features = phon_feat.loc[[p_1], 'advanced_tongue_root':].values
                        feat.append(features)
    if target_word[-1] != 'ː':
        if target_word[-1] not in ignore_phon_lst:
            feat_last_phoneme = phon_feat.loc[[target_word[-1]], 'advanced_tongue_root':].values
            feat.append(feat_last_phoneme)
    feat_lst = [f.tolist() for f in feat]
    return feat_lst


def calculate_features_mini(target_word):
    """
    gets the phoneme features for a word
    :param: target_word: word in its IPA transcription; long phonemes are already merged (e.g., e and : => e:)
    :return: list of phoneme features for the target word
    """
    #   get features for each phoneme
    feat = []
    for phoneme in target_word:
        if phoneme not in ignore_phon_lst:
            features = phon_feat.loc[[phoneme], 'advanced_tongue_root':].values
            feat.append(features)
    feat_lst = [f.tolist() for f in feat]
    return feat_lst


def words_into_syllables(word_1, word_2):
    """
    splits words into their syllables
    :param word_1 in its IPA transcription
    :param word_2 in its IPA transcription
    :return: two lists of syllables
    """
    for item in ['ˌ', 'ˈ', '"']:
        if item in word_1[:]:
            word_1.replace(item, '')
        if item in word_2[:]:
            word_2.replace(item, '')
    syl_lst_w1 = word_1.split('.')
    syl_lst_w2 = word_2.split('.')
    return syl_lst_w1, syl_lst_w2


def word_into_syllables(target_word):
    """
    splits word into its syllables
    :param target_word: in its IPA transcription
    :return: lists of syllables
    """
    for item in ['ˌ', 'ˈ', '"']:
        if item in target_word[:]:
            target_word.replace(item, '')
    syl_lst = target_word.split('.')
    return syl_lst


def same_syllable_number_per_word(syl_lst_w1, syl_lst_w2):
    """
    if one word is longer than the other, we need to fill in empty phonemes aka. all features = 0
    :param syl_lst_w1: list of syllables of word 1
    :param syl_lst_w2: list of syllables of word 2
    :return: two vectors of syllables where the dimensions are the same;
    the shorter word was filled by 'empty', which becomes a vector of 0s later on
    """

    #   calc difference in length and create list of that length
    dl = len(syl_lst_w1) - len(syl_lst_w2)
    to_be_added = ['empty'] * np.abs(dl)

    #   depending on which feature vector is longer, add the 0-vector
    if dl < 0:
        syl_lst_w1_new = syl_lst_w1 + to_be_added
        syl_lst_w2_new = syl_lst_w2

    elif dl > 0:
        syl_lst_w2_new = syl_lst_w2 + to_be_added
        syl_lst_w1_new = syl_lst_w1

    else:
        syl_lst_w1_new = syl_lst_w1
        syl_lst_w2_new = syl_lst_w2

    return syl_lst_w1_new, syl_lst_w2_new


def same_syllable_length(syl_1, syl_2, syl_feature_w1_df, syl_feature_w2_df):
    """
    if one word is longer than the other, we need to fill in empty phonemes aka. all features = 0
    :param syl_feature_w2_df: df with syllable features which occur in w2
    :param syl_feature_w1_df: df with syllable features which occur in w1
    :param syl_1: Syllable i of word 1 with i in (1,...,n), n = #syllables in word 1
    :param syl_2: Syllable j of word 2 with j in (1,...,m), m = #syllables in word 2
    :return: two vectors of phoneme features where the dimensions are the same;
    the shorter syllable was filled by 'empty' features, aka. a vector of 0s
    """

    syl_1_feat = syl_feature_w1_df.loc[syl_1, 'features']
    syl_2_feat = syl_feature_w2_df.loc[syl_2, 'features']

    #   calc difference in length and create list of that length
    # print(syl_1, len(syl_1_feat), syl_2, len(syl_2_feat))
    dl = len(syl_1_feat) - len(syl_2_feat)
    to_be_added = [[[0] * 37]] * np.abs(dl)

    #   depending on which feature vector is longer, add the 0-vector
    if dl < 0:
        syl1_feat_new = syl_1_feat + to_be_added
        syl2_feat_new = syl_2_feat

    elif dl > 0:
        syl2_feat_new = syl_2_feat + to_be_added
        syl1_feat_new = syl_1_feat

    else:
        syl1_feat_new = syl_1_feat
        syl2_feat_new = syl_2_feat

    #   remove dimension that is not needed (as in old version of this function)
    syl1_features = []
    for feat in syl1_feat_new:
        syl1_features.append(feat[0])
    syl2_features = []
    for feat in syl2_feat_new:
        syl2_features.append(feat[0])

    # print(len(syl1_features), len(syl2_features))
    # print(syl_1, 'features after:', syl1_features)
    # print(syl_2, 'features after:', syl2_features)

    return syl1_features, syl2_features


def unique_combinations(w1_features, w2_features):
    """
    :param: w1_features: Phoneme Features of word 1
    :param: w2_features: Phoneme Features of word 2
    :return: a list of unique combinations of phoneme features of both words
    """
    w1_feature_comb = [item for item in permutations(w1_features, len(w1_features))]
    w2_feature_comb = [item for item in permutations(w2_features, len(w2_features))]
    combs = []
    for element1 in w1_feature_comb:
        for element2 in w2_feature_comb:
            combs.append([element1, element2])
    return combs


def get_fd_order(word_features):  # , length='No'):
    """
    :param: the phoneme features of the two input words
    :return: the feature distance of the two input words
    """
    eucl_dists = []
    for i in range(len(word_features[0])):
        # print([np.int_(word_features[0][i]), np.int_(word_features[1][i])])
        eucl_dst = distance.euclidean(np.int_(word_features[0][i]), np.int_(word_features[1][i]))
        # print(eucl_dst)
        eucl_dists.append(eucl_dst)
    #   sum up all distances between phonemes
    feature_distance = sum_numbers(eucl_dists)
    return feature_distance


def get_pd_laing(ipa_lst):
    """
    :param: list with all words in their ipa transcription
    :return: a dataframe with the "feature distance Laing" (FDL) of the two input words
    """
    pd_laing_df = pd.DataFrame(columns=IPA_lst)

    for w1 in ipa_lst:
        w1_dists = []
        #   the syl_feature_w1_df stores the features of the syllables of word w1
        syl_feature_w1_df = pd.DataFrame(columns=['syllable', 'features'])
        #   get list of syllables for word w1
        w1_syl = word_into_syllables(w1)

        #   calculate phoneme features of the word
        w1_syl_feat = []
        for syllable_w1 in w1_syl:
            syl_feat = calculate_features(syllable_w1)
            syl_feature_w1_df.loc[len(syl_feature_w1_df.index)] = [syllable_w1, syl_feat]
            w1_syl_feat.append(syl_feat)
        #   in case we later need an empty phoneme feature, we add an 'phoneme' called 'empty'
        syl_feature_w1_df.loc[len(syl_feature_w1_df.index)] = ['empty', [[[0] * 37]]]
        # syl_feature_w1_df = syl_feature_w1_df.set_index('syllable')
        #   we do not need multiple rows with the same features, so we only keep the first one
        # syl_feature_w1_df_no_duplicates = syl_feature_w1_df.drop_duplicates(keep='first')

        for w2 in ipa_lst:
            dist_lst_syllables = []
            #   the syl_feature_w2_df stores the features of the syllables of word w2
            syl_feature_w2_df = pd.DataFrame(columns=['syllable', 'features'])
            #   get list of syllables for word w2
            w2_syl = word_into_syllables(w2)

            w2_syl_feat = []
            for syllable_w2 in w2_syl:
                syl_feat = calculate_features(syllable_w2)
                syl_feature_w2_df.loc[len(syl_feature_w2_df.index)] = [syllable_w2, syl_feat]
                w2_syl_feat.append(syl_feat)
            syl_feature_w2_df.loc[len(syl_feature_w2_df.index)] = ['empty', [[[0] * 37]]]
            # syl_feature_w2_df = syl_feature_w2_df.set_index('syllable')
            # syl_feature_w2_df_no_duplicates = syl_feature_w2_df.drop_duplicates(keep='first')

            #   get same number of syllables per word
            w1_syllables, w2_syllables = same_syllable_number_per_word(w1_syl, w2_syl)

            #   now we need to check the position of the vowel in the syllables
            for syllable_w1, syllable_w2 in zip(w1_syllables, w2_syllables):
                syllable_phonemes_w1 = []
                syllable_phonemes_w2 = []
                #   for this, we first need to combine the colon and the phoneme before to one phoneme
                for p_1, p_2 in zip(syllable_w1, syllable_w1[1:]):
                    if p_1 not in ignore_phon_lst:
                        #   check if a colon follows a phoneme; if so, put them together as one phoneme
                        if not contains_letter(p_1, 'ː'):
                            if p_2 == 'ː':
                                p1 = "".join((p_1, p_2))
                            else:
                                p1 = p_1
                        else:
                            pass
                        syllable_phonemes_w1.append(p1)
                if syllable_w1[-1] != 'ː':
                    if syllable_w1[-1] not in ignore_phon_lst:
                        syllable_phonemes_w1.append(syllable_w1[-1])
                # print(syllable_phonemes_w1)

                for p_1, p_2 in zip(syllable_w2, syllable_w2[1:]):
                    if p_1 not in ignore_phon_lst:
                        #   check if a colon follows a phoneme; if so, put them together as one phoneme
                        if not contains_letter(p_1, 'ː'):
                            if p_2 == 'ː':
                                p1 = "".join((p_1, p_2))
                            else:
                                p1 = p_1
                        else:
                            pass
                        syllable_phonemes_w2.append(p1)
                if syllable_w2[-1] != 'ː':
                    if syllable_w2[-1] not in ignore_phon_lst:
                        syllable_phonemes_w2.append(syllable_w2[-1])
                # print(syllable_phonemes_w2)

                #   now we get the vowel positions of the syllables in both words
                vowel_position_word_1 = 0
                joined_syllable = ''.join(syllable_phonemes_w1)
                if joined_syllable == 'empty':
                    syllable_phonemes_w1 = ['empty', 'empty', 'empty', 'empty', 'empty']
                for phoneme_w1 in syllable_phonemes_w1:
                    if phoneme_w1 not in ignore_phon_lst:
                        if phon_feat.loc[phoneme_w1, 'segment_class'] == 'c':
                            vowel_position_word_1 += 1
                        elif phon_feat.loc[phoneme_w1, 'segment_class'] == 'v':
                            break
                        else:
                            vowel_position_word_1 = 0
                            break

                # print('vowel position %s: ' % w1, vowel_position_word_1)

                vowel_position_word_2 = 0
                joined_syllable = ''.join(syllable_phonemes_w2)
                if joined_syllable == 'empty':
                    syllable_phonemes_w2 = ['empty', 'empty', 'empty', 'empty', 'empty']
                for phoneme_w2 in syllable_phonemes_w2:
                    if phoneme_w2 not in ignore_phon_lst:
                        if phon_feat.loc[phoneme_w2, 'segment_class'] == 'c':
                            vowel_position_word_2 += 1
                        elif phon_feat.loc[phoneme_w2, 'segment_class'] == 'v':
                            break
                        else:
                            vowel_position_word_2 = 0
                            break
                # print('vowel position %s: ' % w2, vowel_position_word_2)
                max_vowel_position = max(vowel_position_word_1, vowel_position_word_2)

                #   now we have the higher vowel position of the two syllables of the words
                #   as we still have many 'empty's for empty syllables, we need to reduce them to 1 'empty'
                if 'empty' in syllable_phonemes_w1:
                    syllable_phonemes_w1 = ['empty']
                if 'empty' in syllable_phonemes_w2:
                    syllable_phonemes_w2 = ['empty']
                # print(syllable_phonemes_w1, syllable_phonemes_w2)
                #   check if word 1 has a vowel at the max vowel position
                if vowel_position_word_1 == max_vowel_position:
                    if vowel_position_word_2 == max_vowel_position:
                        #   same vowel position checked, now check if the syllables are of same length
                        while len(syllable_phonemes_w1) != len(syllable_phonemes_w2):
                            if len(syllable_phonemes_w1) < len(syllable_phonemes_w2):
                                syllable_phonemes_w1.append('empty')
                            elif len(syllable_phonemes_w2) < len(syllable_phonemes_w1):
                                syllable_phonemes_w2.append('empty')
                    else:
                        if syllable_phonemes_w2 == ['empty']:
                            pass
                        else:
                            while vowel_position_word_2 != max_vowel_position:
                                #   it's not the same vowel position, so we have to add an empty phoneme
                                #   to the beginning of the syllable
                                syllable_phonemes_w2.insert(0, 'empty')
                                vowel_position_word_2 = 0
                                for phoneme_w2 in syllable_phonemes_w2:
                                    if phoneme_w2 not in ignore_phon_lst:
                                        if phon_feat.loc[phoneme_w2, 'segment_class'] == 'c':
                                            vowel_position_word_2 += 1
                                        elif phon_feat.loc[phoneme_w2, 'segment_class'] == 'v':
                                            break
                                        else:
                                            vowel_position_word_2 += 1
                        while len(syllable_phonemes_w1) != len(syllable_phonemes_w2):
                            if len(syllable_phonemes_w1) < len(syllable_phonemes_w2):
                                syllable_phonemes_w1.append('empty')
                            elif len(syllable_phonemes_w2) < len(syllable_phonemes_w1):
                                syllable_phonemes_w2.append('empty')
                else:
                    if syllable_phonemes_w1 == ['empty']:
                        pass
                    else:
                        while vowel_position_word_1 != max_vowel_position:
                            syllable_phonemes_w1.insert(0, 'empty')
                            vowel_position_word_1 = 0
                            for phoneme_w1 in syllable_phonemes_w1:
                                if phoneme_w1 not in ignore_phon_lst:
                                    if phon_feat.loc[phoneme_w1, 'segment_class'] == 'c':
                                        vowel_position_word_1 += 1
                                    elif phon_feat.loc[phoneme_w1, 'segment_class'] == 'v':
                                        break
                                    else:
                                        vowel_position_word_1 += 1
                    while len(syllable_phonemes_w1) != len(syllable_phonemes_w2):
                        if len(syllable_phonemes_w1) < len(syllable_phonemes_w2):
                            syllable_phonemes_w1.append('empty')
                        elif len(syllable_phonemes_w2) < len(syllable_phonemes_w1):
                            syllable_phonemes_w2.append('empty')

                # print('syllables phonemes w1: ', syllable_phonemes_w1,
                # 'syllables phonemes w2: ', syllable_phonemes_w2)
                features_w1_syl = calculate_features_mini(syllable_phonemes_w1)
                features_w2_syl = calculate_features_mini(syllable_phonemes_w2)
                # print(syllable_phonemes_w1, syllable_phonemes_w2)
                # print(features_w1_syl, features_w2_syl)
                # print('features w1 syl: ', len(features_w1_syl), 'features w2 syl: ', len(features_w2_syl))
                pd_syl_w1w2 = get_fd_order([features_w1_syl, features_w2_syl])  # , length='No')
                dist_lst_syllables.append(pd_syl_w1w2)
            w1_w2_dist = sum(dist_lst_syllables)
            # pd_word_pair_normalised = w1_w2_dist / max(len(w1), len(w2))
            w1_dists.append(w1_w2_dist)
            print('dist between syllables of %s and %s' % (w1, w2), w1_w2_dist)
        pd_laing_df.loc[len(pd_laing_df)] = w1_dists
    return pd_laing_df


def syllable_ovc(syllable):
    """
    In this function we split a syllable in its onset, vowel and coda.
    :param syllable: a list which contains all phonemes of a syllable
    :return: a list with 3 elements, i.e., one element is onset, one is vowel and one is coda
    """
    vowel_index = 100
    vowel_index_2 = 100
    consonant_index = 100
    for phoneme in syllable:
        if phoneme in ignore_phon_lst:
            pass
        else:
            if phon_feat.loc[phoneme, 'segment_class'] == 'v':
                if vowel_index == 100:
                    vowel_index = syllable.index(phoneme)
                    for phon in syllable[vowel_index + 1:]:
                        if phon not in ignore_phon_lst:
                            if phon_feat.loc[phon, 'segment_class'] == 'c':
                                consonant_index = vowel_index + syllable[vowel_index + 1:].index(phon)
                                #   what if the same phoneme occurs more than once?
                            else:
                                vowel_index_2 = syllable.index(phoneme) + syllable[vowel_index + 1:].index(phon)
                else:
                    vowel_index_2 = syllable.index(phoneme)
                    for phon in syllable[vowel_index_2 + 1:]:
                        if phon not in ignore_phon_lst:
                            if phon_feat.loc[phon, 'segment_class'] == 'c':
                                consonant_index = vowel_index_2 + syllable[vowel_index + 1:].index(phon)
                                #   what if the same phoneme occurs more than once?
                            else:
                                vowel_index_3 = vowel_index_2 + syllable[vowel_index + 1:].index(phon)
            else:
                pass

    if consonant_index == 100:
        if phon_feat.loc[syllable[0], 'segment_class'] == 'c':
            onset = syllable[:vowel_index]
            vowel = syllable[vowel_index:]
            coda = []
            syllable_ovc_lst = [onset, vowel, coda]
            return syllable_ovc_lst
        else:
            onset = []
            vowel = syllable[:]
            coda = []
            syllable_ovc_lst = [onset, vowel, coda]
            return syllable_ovc_lst
    else:
        if vowel_index_2 == 100:
            onset = syllable[0:vowel_index]
            vowel = syllable[vowel_index:vowel_index + 1]
            coda = syllable[vowel_index + 1:]
            syllable_ovc_lst = [onset, vowel, coda]
            return syllable_ovc_lst
        else:
            onset = syllable[0:vowel_index]
            vowel = syllable[vowel_index:vowel_index + 2]
            coda = syllable[vowel_index + 2:]
            syllable_ovc_lst = [onset, vowel, coda]
            return syllable_ovc_lst

    # print(syllable, 'onset: ', onset, 'vowel: ', vowel, 'coda: ', coda)


def get_long_phonemes(syllable):
    """
    In this function we loop through all phonemes and store a long phoneme, i.e., one with a colon following,
    in the list with all phonemes.
    :param syllable: a list which contains all phonemes of a syllable
    :return: a list with all phonemes of the syllable
    """
    syllable_phonemes = []
    for p_1, p_2 in zip(syllable, syllable[1:]):
        if p_1 not in ignore_phon_lst:
            #   check if a colon follows a phoneme; if so, put them together as one phoneme
            if not contains_letter(p_1, 'ː'):
                if p_2 == 'ː':
                    p1 = "".join((p_1, p_2))
                    syllable_phonemes.append(p1)
                else:
                    p1 = p_1
                    syllable_phonemes.append(p1)
            else:
                pass
    if syllable[-1] != 'ː':
        if syllable[-1] not in ignore_phon_lst:
            syllable_phonemes.append(syllable[-1])
    return syllable_phonemes


def distance_syll_part(s1_onset, s2_onset):
    if len(s1_onset) == len(s2_onset):
        if len(s1_onset) == 0:
            #   if the syllable part is empty
            dist_onset_s1_s2 = 0
            return dist_onset_s1_s2
        s1_onset_feat = calculate_features_mini(s1_onset)
        s2_onset_feat = calculate_features_mini(s2_onset)
        if len(s1_onset) == 1:
            dist_onset_s1_s2 = get_fd_order([s1_onset_feat, s2_onset_feat])
            return dist_onset_s1_s2
            # distances_comb.append(dist_onset_s1_s2)
            # print('dist_s1_s2: ', dist_s1_s2)
        elif len(s1_onset) == 2:
            #   onset word1 = (a,b), onset word2 = (x,y)
            #   pd(ab,xy) = min[(pd(a,x)+pd(b,y)), (pd(none,x)+pd(a,y)+pd(b,none)), (pd,(none, a)+pd(x,b)+pd(y,none))]
            comb_1 = get_fd_order([[s1_onset_feat[0], s1_onset_feat[1]], [s2_onset_feat[0], s2_onset_feat[1]]])
            comb_2 = get_fd_order(
                [[[[0] * 37], s1_onset_feat[0], s1_onset_feat[1]], [s2_onset_feat[0], s2_onset_feat[1], [[0] * 37]]])
            comb_3 = get_fd_order(
                [[[[0] * 37], s2_onset_feat[0], s2_onset_feat[1]], [s1_onset_feat[0], s1_onset_feat[1], [[0] * 37]]])
            dist_onset_s1_s2 = min(comb_1, comb_2, comb_3)
            return dist_onset_s1_s2
            # distances_comb.append(dist_onset_s1_s2)
        elif len(s1_onset) == 3:
            #   onset word1 = (a,b,c), onset word2 = (x,y,c)
            #   pd(abc,xyz) = pd(a,x)+pd(b,y)+pd(c,z)
            dist_onset_s1_s2 = get_fd_order([s1_onset_feat, s2_onset_feat])
            # distances_comb.append(dist_onset_s1_s2)
            return dist_onset_s1_s2
    else:
        #   determine the longer and shorter word
        if len(s1_onset) > len(s2_onset):
            longer_syl = s1_onset
            shorter_syl = s2_onset
        else:
            longer_syl = s2_onset
            shorter_syl = s1_onset

        #   calculate distances
        if len(shorter_syl) == 0:
            #   onset word1 = (a), onset word2 = ()
            #   pd(a,none) = pd(a,null vector)
            longer_syl_onset_feat = calculate_features_mini(longer_syl)
            shorter_syl_onset_feat = calculate_features_mini(['empty'] * len(longer_syl))
            dist_onset_s1_s2 = get_fd_order([longer_syl_onset_feat, shorter_syl_onset_feat])
            return dist_onset_s1_s2
            # distances_comb.append(dist_onset_s1_s2)
        elif len(longer_syl) == 2:
            #   onset word1 = (a,b), onset word2 = (x)
            #   pd(ab,x) = pd(a,x)+pd(b,x)
            longer_syl_onset_feat = calculate_features_mini(longer_syl)
            shorter_syl_onset_feat = calculate_features_mini(shorter_syl)
            dist_onset_s1_s2 = get_fd_order([[longer_syl_onset_feat[0], longer_syl_onset_feat[1]],
                                             [shorter_syl_onset_feat[0], shorter_syl_onset_feat[0]]])
            return dist_onset_s1_s2
            #   should I add the mean here and not just sum the dists up?
            # distances_comb.append(dist_onset_s1_s2)
        elif len(longer_syl) == 3:
            longer_syl_onset_feat = calculate_features_mini(longer_syl)
            shorter_syl_onset_feat = calculate_features_mini(shorter_syl)
            if len(shorter_syl) == 2:
                #   onset word1 = (a,b,c), onset word2 = (x,y)
                #   pd(abc,xy) = min( [pd(ab,xy)+min(pd(c,x),pd(c,y))],
                #                     [pd(ac,xy)+min(pd(b,x),pd(b,y))],
                #                     [pd(bc,xy)+min(pd(a,x),pd(a,y))]  )
                comb_1 = get_fd_order([[longer_syl_onset_feat[0], longer_syl_onset_feat[1]],
                                       [shorter_syl_onset_feat[0], shorter_syl_onset_feat[1]]]) + \
                         min(get_fd_order([[longer_syl_onset_feat[2]], [shorter_syl_onset_feat[0]]]),
                             get_fd_order([[longer_syl_onset_feat[2]], [shorter_syl_onset_feat[1]]]))
                comb_2 = get_fd_order([[longer_syl_onset_feat[0], longer_syl_onset_feat[2]],
                                       [shorter_syl_onset_feat[0], shorter_syl_onset_feat[1]]]) + \
                         min(get_fd_order([[longer_syl_onset_feat[1]], [shorter_syl_onset_feat[0]]]),
                             get_fd_order([[longer_syl_onset_feat[1]], [shorter_syl_onset_feat[1]]]))
                comb_3 = get_fd_order([[longer_syl_onset_feat[1], longer_syl_onset_feat[2]],
                                       [shorter_syl_onset_feat[0], shorter_syl_onset_feat[1]]]) + \
                         min(get_fd_order([[longer_syl_onset_feat[0]], [shorter_syl_onset_feat[0]]]),
                             get_fd_order([[longer_syl_onset_feat[0]], [shorter_syl_onset_feat[1]]]))
                dist_onset_s1_s2 = min(comb_1, comb_2, comb_3)
                return dist_onset_s1_s2
            else:
                #   onset word1 = (a,b,c), onset word2 = (x)
                #   pd(ab,x) = pd(a,x)+pd(b,x)+pd(c,x)
                dist_onset_s1_s2 = get_fd_order([[longer_syl_onset_feat[0], longer_syl_onset_feat[1],
                                                  longer_syl_onset_feat[2]],
                                                 [shorter_syl_onset_feat[0], shorter_syl_onset_feat[0],
                                                  shorter_syl_onset_feat[0]]])
                return dist_onset_s1_s2
            #   should I add the mean here and not just sum the dists up?
            # distances_comb.append(dist_onset_s1_s2)
        # return dist_onset_s1_s2


def compare_ovc(syllable_combination):
    """
    In this function, we give a pair of syllables which we want to compare. The function calculates the feature
    distances between onset, vowel and coda, including all possible slot changes within them,
    sums them up and returns the minimum distance of this syllable combination.
    :param syllable_combination: a list of two syllables from two different words
    :return: the minimum feature distance between the two syllables
    """
    # print('syllable comb: ', syllable_combination)
    distances_comb = []
    for comb in syllable_combination:
        # print('comb: ', comb)
        s1 = comb[0]
        s2 = comb[1]
        # print(s1, s2)

        #   compare onset
        if s1 is None:
            s1_onset = []
        else:
            s1_onset = s1[0]
        if s2 is None:
            s2_onset = []
        else:
            s2_onset = s2[0]
        onset_dist = distance_syll_part(s1_onset, s2_onset)

        #   compare vowel
        s1_vowel = s1[1]
        s2_vowel = s2[1]
        vowel_dist = distance_syll_part(s1_vowel, s2_vowel)

        #   compare coda
        s1_coda = s1[2]
        s2_coda = s2[2]
        coda_dist = distance_syll_part(s1_coda, s2_coda)

        # print(onset_dist, vowel_dist, coda_dist)
        comb_distance = onset_dist + vowel_dist + coda_dist
        distances_comb.append(comb_distance)
    comb_distance = sum(distances_comb)
    return comb_distance


def get_pd_kalinowski(ipa_lst):
    """
    In this phonological distance measurement we split syllables in onset, vowel and coda. Within each of them, we
    move the vowels to find the smallest phonological distance between two words.
    :param: list with all words in their ipa transcription
    :return: the feature distance Kalinowski (FDK) of the two input words
    """
    pd_kalinowski_df = pd.DataFrame([])
    for w1 in ipa_lst:
        w1_dists = []
        #   split word into its syllables -> get list of syllables
        w1_syls = word_into_syllables(w1)
        #   get the onset, vowel and coda of syllable
        ovc_syllables_w1 = []
        for syllable in w1_syls:
            syllable_phonemes = get_long_phonemes(syllable)
            syllable_ovc_output = syllable_ovc(syllable_phonemes)
            ovc_syllables_w1.append(syllable_ovc_output)
        # print('ovc_sylls_w1: ', ovc_syllables_w1)
        for w2 in ipa_lst:
            # print('w2: ', w2)
            #   split word into its syllables -> get list of syllables
            w2_syls = word_into_syllables(w2)
            # print('w2_syls: ', w2_syls)
            #   get the onset, vowel and coda of syllable
            ovc_syllables_w2 = []
            for syllable in w2_syls:
                syllable_phonemes = get_long_phonemes(syllable)
                # print('syllables phonemes: ', syllable_phonemes)
                syllable_ovc_output = syllable_ovc(syllable_phonemes)
                # print('syllable_ovc_output: ', syllable_ovc_output)
                ovc_syllables_w2.append(syllable_ovc_output)
            #   now we have to compare all syllables with each other
            # print('ovc_sylls_w2: ', ovc_syllables_w2)
            #   get shorter and longer word so that the calculation is always in the same order
            #   otherwise we will get two different pds for pd(w1,w2) and pd(w2,w1)
            if len(ovc_syllables_w1) > len(ovc_syllables_w2):
                shorter_word_syls = ovc_syllables_w2
                longer_word_syls = ovc_syllables_w1
            elif len(ovc_syllables_w1) < len(ovc_syllables_w2):
                shorter_word_syls = ovc_syllables_w1
                longer_word_syls = ovc_syllables_w2
            else:
                phon_no_w1 = []
                for syl in ovc_syllables_w1:
                    if syl is None:
                        pass
                    else:
                        for phon in syl:
                            phon_no_w1.append(phon)
                phon_no_w2 = []
                for syl in ovc_syllables_w2:
                    if syl is None:
                        pass
                    else:
                        for phon in syl:
                            phon_no_w2.append(phon)
                if len(phon_no_w1) > len(phon_no_w2):
                    shorter_word_syls = ovc_syllables_w2
                    longer_word_syls = ovc_syllables_w1
                elif len(phon_no_w1) < len(phon_no_w2):
                    shorter_word_syls = ovc_syllables_w1
                    longer_word_syls = ovc_syllables_w2
                else:
                    shorter_word_syls = ovc_syllables_w1
                    longer_word_syls = ovc_syllables_w2

            #   get permutations of syllables of w1 and w2
            #   create empty list to store the combinations
            syls_combinations_lst = []
            #   Get all permutations of ovc_syllables_w1 with length of ovc_syllables_w2
            # print('longer_word_syls: ', longer_word_syls, 'shorter word syls: ', shorter_word_syls)
            permut = itertools.permutations(longer_word_syls)

            #   zip() is called to pair each permutation and shorter list element into combination
            for comb in permut:
                zipped = zip(comb, shorter_word_syls)
                syls_combinations_lst.append(list(zipped))

            # print(syls_combinations_lst)
            #   within the permutation, calculate the pd of the syllables:
            combination_dists = []
            for combination in syls_combinations_lst:
                # print(ovc_syllables_w1, ovc_syllables_w2, combination)
                comb_dist = compare_ovc(combination)

                #   if the two words have a different number of syllables, we only included n-m syllables.
                #   We now have to include the others.

                included_syllables = []
                excluded_syllables = []

                if len(ovc_syllables_w1) != len(ovc_syllables_w2):
                    for syllable in longer_word_syls:
                        if syllable not in combination:
                            excluded_syllables.append(syllable)
                    missing_sylls_dist_lst = []
                    for missing_syl in excluded_syllables:
                        missing_syl_dist_lst = []
                        for shorter_word_syl in shorter_word_syls:
                            # print('missing: ', missing_syl, 'shorter word syll: ', shorter_word_syl)
                            syll_dist = compare_ovc([(missing_syl, shorter_word_syl)])
                            missing_syl_dist_lst.append(syll_dist)
                        missing_syll_pd = np.mean(missing_syl_dist_lst)
                        missing_sylls_dist_lst.append(missing_syll_pd)
                    missing_sylls_dist = sum(missing_syl_dist_lst)
                #   ADD MISSING SYLLS
                    comb_dist_full = comb_dist + missing_sylls_dist
                else:
                    comb_dist_full = comb_dist
                combination_dists.append(comb_dist_full)
            w1_w2_dist_raw = min(combination_dists)
            w1_w2_dist = w1_w2_dist_raw  # / max(len(w1), len(w2))
            w1_dists.append(w1_w2_dist)
            #   append list of PD of w1 with all other words to df
        if w1 in pd_kalinowski_df.columns:
            if (w1 + "_2") in pd_kalinowski_df.columns:
                pd_kalinowski_df[w1 + "_3"] = w1_dists
            else:
                pd_kalinowski_df[w1 + "_2"] = w1_dists
        else:
            pd_kalinowski_df[w1] = w1_dists

    return pd_kalinowski_df


def levenshtein_distance(ipa_lst):
    """
    Gives you a matrix with pairwise levenshtein distances of the words in ipa_lst. The difference to the
    levenshtein distance from the levenshtein package is that it includes ":" which makes a phoneme longer,
    but does not add a new phoneme.
    :param ipa_lst: list of words in its ipa transcription
    :return: a matrix with pairwise levenshtein distances of the words in ipa_lst
    """
    ld_df = pd.DataFrame(columns=ipa_lst, index=ipa_lst)
    for word_1 in ipa_lst:
        w_1 = get_long_phonemes(word_1)
        for word_2 in ipa_lst:
            length = 0
            w_2 = get_long_phonemes(word_2)
            if len(w_1) == len(w_2):
                longer_word = w_1
                shorter_word = w_2
            elif len(w_1) > len(w_2):
                longer_word = w_1
                shorter_word = w_2
            else:
                longer_word = w_2
                shorter_word = w_1
            while len(longer_word) != len(shorter_word):
                shorter_word.append('_')
            for p1, p2 in zip(longer_word, shorter_word):
                if p1 == p2:
                    pass
                else:
                    length += 1
            ld_df.at[word_1, word_2] = length
    return ld_df


def normalise(df):
    max_value = df.to_numpy().max()
    print('max value: ', max_value)
    min_value = df.to_numpy().min()
    print('min value: ', min_value)
    norm_df = (df - min_value) / (max_value - min_value)
    return norm_df


def max_length(ipa_lst):
    """
    Get the maximum length of two words.
    :param ipa_lst: list of words in their IPA writing.
    :return: df of pairwise maximum lengths
    """
    max_length_dataframe = pd.DataFrame(columns=ipa_lst, index=ipa_lst)
    for w1 in ipa_lst:
        w1_phon = get_long_phonemes(w1)
        for w2 in ipa_lst:
            w2_phon = get_long_phonemes(w2)
            max_length_dataframe.at[w1, w2] = max(len(w1_phon), len(w2_phon))
    return max_length_dataframe


"""""""""""""""""
PART 2:
USE FUNCTIONS
"""""""""""""""""

#   files to import
words = pd.read_csv('./csv_files/words.csv', sep=";", header=int())
words_lst = words.definition_new.tolist()
missing_words = pd.read_csv('./csv_files/missing_words_karen.csv', sep=";", header=int())
phon_feat = pd.read_csv('./csv_files/phon_feat_nor.csv', sep=";", header=int(), index_col='segment')

items_df_IPA_nlb = pd.DataFrame(words, columns=['num_item_id', 'IPA_nlb'])
items_df_IPA_nlb.index = np.arange(0, len(items_df_IPA_nlb))

ignore_phon_lst = ['ˌ', 'ˈ', '.', '"']

#   in the following, we work on the IPA transcriptions of the
#   IPA dataframe so that we can use it for the levenshtein distance

#   do some needed preparatory work on the IPA transcriptions, and then get a list od IPA words
n = 0
for row in range(len(items_df_IPA_nlb)):
    if pd.isna(items_df_IPA_nlb.at[row, 'IPA_nlb']):
        items_df_IPA_nlb.at[row, 'IPA_nlb'] = missing_words.loc[n, 'missing ipa']
        n += 1
IPA_lst = items_df_IPA_nlb.IPA_nlb.tolist()

#   delete space in IPA transcriptions of words
for idx, ele in enumerate(IPA_lst):
    IPA_lst[idx] = ele.replace(' ', '')

"""""""""""""""""""""""""""""""""""""""""""""
0. Get df of the maximum length of word pairs
"""""""""""""""""""""""""""""""""""""""""""""

max_length_df = max_length(IPA_lst)
max_length_df.to_csv(save_to_csv + 'max_length_df.csv')

"""""""""""""""""""""""
1. Levenshtein distance
"""""""""""""""""""""""

''' in the following, we calculate the levenshtein distance and
normalise it by dividing by max(len(word1), len(word2)) '''

LEVENSHTEIN = True

if LEVENSHTEIN:
    lev_dist = levenshtein_distance(IPA_lst)
    lev_dist.to_csv(save_to_csv + 'LD_no_length.csv')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
2. Get data frame of phoneme features
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#   there is a problem with a few phonemes, which we decided to replace
#   with a different phoneme which is close to it
IPA_lst[522] = '"lɪː.tn'
IPA_lst[532] = 'ˈɛ.sn'
IPA_lst[535] = '"ʃɪ.tn'
IPA_lst[544] = '"sʉl.tn'
IPA_lst[568] = '"moː.ɳ'
IPA_lst[644] = '"ʉː.tn.fɔr'
IPA_lst[645] = '"ʉː.tn.poː'
IPA_lst[84] = 'plɑsː.ti.ˈli.nɑˈ'
IPA_lst[231] = 'ˈçœ.kːən.rʉlːˈ'
IPA_lst[565] = 'i.ˈkvɛlːˈ'

#   get all possible pairs of words in the words list
word_pairs = [[word1, word2] for idx, word1 in enumerate(IPA_lst) for word2 in IPA_lst[idx + 1:]]
ignore_phon_lst = ['ˌ', 'ˈ', '.', '"', '.']

#   get the phoneme features of every word (to do so, set CALCULATE_FEATURES = True)

feature_df = pd.DataFrame(columns=['features'])
for word in IPA_lst:
    output_features_raw = calculate_features(word)
    feature_df.loc[len(feature_df.index), 'features'] = output_features_raw

feature_df = feature_df.set_index([IPA_lst])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. euclidean distance between phonological features with vowel alignment (FDL)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#   calculate the Feature Distances Laing (to do so, set FEATURE_DISTANCE_LAING = True)

FEATURE_DISTANCE_LAING = True
if FEATURE_DISTANCE_LAING:
    output_FDL = get_pd_laing(IPA_lst[:20])
    output_FDL = output_FDL.set_axis(IPA_lst, axis='index')
    output_FDL = output_FDL.set_axis(IPA_lst, axis=1)
    output_FDL.to_csv(save_to_csv + 'FDL_no_length_eucl_test.csv')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
4. euclidean distance between phonological features by Kalinowski (FDK)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#   calculate the Feature Distances Kalinowski (to do so, set FEATURE_DISTANCE_Kalinowski = True)

FEATURE_DISTANCE_KALINOWSKI = True

if FEATURE_DISTANCE_KALINOWSKI:
    output_FDM = get_pd_kalinowski(IPA_lst)
    output_FDM = output_FDM.set_axis(IPA_lst, axis='index')
    output_FDM = output_FDM.set_axis(IPA_lst, axis=1)
    output_FDM.to_csv(save_to_csv + 'FDK_no_length_eucl.csv')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
5. divide all distance dfs by the length of the longer word of the word pair
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

LENGTH = True

FDL_no_length = pd.read_csv('./csv_outputs/FDL_no_length_eucl.csv', sep=",", header=int(), index_col=0)
LD_no_length = pd.read_csv('./csv_outputs/LD_no_length.csv', sep=",", header=int(), index_col=0)
FDK_no_length = pd.read_csv('./csv_outputs/FDK_no_length_eucl.csv', sep=",", header=int(), index_col=0)

if LENGTH:
    array = np.array(max_length_df)
    inv = 1. / array
    inv_matrix = pd.DataFrame(inv, columns=IPA_lst, index=IPA_lst)
    count = 0
    for df in [FDL_no_length, LD_no_length, FDK_no_length]:
        if count == 0:
            string = 'FDL_length_eucl'
        elif count == 1:
            string = 'LD_length_eucl'
        else:
            string = 'FDK_length_eucl'
        df_length = pd.DataFrame(df.values * inv_matrix.values, columns=IPA_lst, index=IPA_lst)
        df_length.to_csv(save_to_csv + string + '.csv')
        count += 1

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
6. normalise all distance dataframes to make them comparable
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
NORMALISE_DF = True

if NORMALISE_DF:
    LD = pd.read_csv('./csv_outputs/LD_length_eucl.csv', sep=",", header=int(), index_col=0)
    FDL = pd.read_csv('./csv_outputs/FDL_length_eucl.csv', sep=",", header=int(), index_col=0)
    FDK = pd.read_csv('./csv_outputs/FDK_length_eucl.csv', sep=",", header=int(), index_col=0)

    LD_norm = normalise(LD)
    LD_norm = LD_norm.set_axis(IPA_lst, axis='index')
    LD_norm = LD_norm.set_axis(IPA_lst, axis=1)
    LD_norm.to_csv(save_to_csv + 'LD_norm.csv')

    FDL_norm = normalise(FDL)
    FDL_norm = FDL_norm.set_axis(IPA_lst, axis='index')
    FDL_norm = FDL_norm.set_axis(IPA_lst, axis=1)
    FDL_norm.to_csv(save_to_csv + 'FDL_norm_eucl.csv')

    FDK_norm = normalise(FDK)
    FDK_norm = FDK_norm.set_axis(IPA_lst, axis='index')
    FDK_norm = FDK_norm.set_axis(IPA_lst, axis=1)
    FDK_norm.to_csv(save_to_csv + 'FDK_norm_eucl.csv')
