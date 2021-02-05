import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
import itertools
import difflib

import matplotlib.pyplot as plt
import numpy as np


# Get data from file
chinese = []  # list of rach chinese sentence
trans1 = []  # list of each 1st translation
trans2 = []  # list of each 2nd translation (machine or human)
score = []  # list of each translation score
# list indicating if each item is a machine or no with bool (for train)
machine_or_no = []
# list indicating if each item is a machine or no with int (for train)
machine_or_no_int = []
# a list indicating the simmilarity of 2 translations
simm = []

filename = "test.txt"

# Open training text file and get data
fp = open(filename, encoding="utf8")
for i, line in enumerate(fp):
    line = line.strip('\n')
    if i % 6 == 0:
        chinese.append(line)
    elif i % 6 == 1:
        trans1.append(line)
    elif i % 6 == 2:
        trans2.append(line)
    elif i % 6 == 3:
        score.append(float(line))
    elif i % 6 == 4:
        if line == "M":
            machine_or_no.append(True)
        else:
            machine_or_no.append(False)
fp.close()


# Calculate the simmilarity semantically with synsets and of the sentence structure using pos_tag
count = 0
for i in range(len(machine_or_no)):
    sent1 = trans1[i]
    sent2 = trans2[i]

    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}

    # list of tuples with word and part of speech in each tuple, len(sent1) tuples
    s1 = nltk.pos_tag(nltk.word_tokenize(sent1))
    s1_pos = []
    for tup in s1:
        s1_pos.append(tup[1])

    s2 = nltk.pos_tag(nltk.word_tokenize(sent2))
    s2_pos = []
    for tup in s2:
        s2_pos.append(tup[1])

    # the parts of speech of one translation matched with another with smart sequence matcher
    sm = difflib.SequenceMatcher(None, s1_pos, s2_pos)
    res = sm.ratio()

    machine_or_no_int.append(int(machine_or_no[i]))

    # Simmilarity between the sentences using synsets...
    s1 = dict(filter(lambda x: len(x[1]) > 0,
                     map(lambda row: (row[0], wn.synsets(
                         row[0],
                         tag_dict[row[1][0]])) if row[1][0] in tag_dict.keys()
                         else (row[0], []), s1)))

    s2 = nltk.pos_tag(nltk.word_tokenize(sent2))

    s2 = dict(filter(lambda x: len(x[1]) > 0,
                     map(lambda row: (row[0], wn.synsets(
                         row[0],
                         tag_dict[row[1][0]])) if row[1][0] in tag_dict.keys()
                         else (row[0], []), s2)))

    resi = {}
    for w2, gr2 in s2.items():
        for w1, gr1 in s1.items():
            tmp = pd.Series(list(map(lambda row: row[1].path_similarity(row[0]),
                                     itertools.product(gr1, gr2)))).dropna()
            if len(tmp) > 0:
                resi[(w1, w2)] = tmp.max()

    similarity = pd.Series(resi).groupby(level=0).max().mean()

    # Average the 2 metrics
    wow = (similarity + res) / 2
    simm.append(wow)

    # draw a line at .65, and if above, it is predicted to be a human
    # add 1 to count if predicted correctly.
    border = .65
    if (wow) > border and not machine_or_no[i]:
        count += 1
    elif wow <= border and machine_or_no[i]:
        count += 1

# accuracy
print(count/len(machine_or_no))
