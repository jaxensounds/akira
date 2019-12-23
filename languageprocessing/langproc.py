# Akira source

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as funct
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import time

# Check for CUDA
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# region Load & preprocess data
timestart = time.perf_counter_ns() / 1e+6
# Load corpus
corpus_name = 'cornell movie-dialogs corpus'
corpus = os.path.join('data', corpus_name)


def println(file, n=10):
    with open(file, 'rb') as datafl:
        lns = datafl.readlines()
    for ln in lns[:n]:
        print(ln)


# Format datafile
# Split each line into fields dictionary
def loadlns(filename, fields):
    lines = {}
    with open(filename, 'r', encoding='iso-8859-1') as file:
        for line in file:
            values = line.split(' +++$+++ ')
            # Extract fields
            lineobject = {}
            for index, field in enumerate(fields):
                lineobject[field] = values[index]
            lines[lineobject['lineID']] = lineobject
    return lines


# Group fields of lines into conversations based on *movie_conversations.txt*
def loadconv(filename, lines, fields):
    conversations = []
    with open(filename, 'r', encoding='iso-8859-1') as file:
        for line in file:
            values = line.split(' +++$+++ ')
            # Extract fields
            conversationalobject = {}
            for index, field in enumerate(fields):
                conversationalobject[field] = values[index]
            # Convert string to list
            utterancepattern = re.compile('L[0-9]+')
            lineids = utterancepattern.findall(conversationalobject['utteranceIDs'])
            # Reassemble lines
            conversationalobject['lines'] = []
            for lineid in lineids:
                conversationalobject['lines'].append(lines[lineid])
                conversations.append(conversationalobject)
    return conversations


# Extract pairs of sentences from conversations
def extractsentencepairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all lines n conversations
        for i in range(len(conversation['lines']) - 1):
            inputln = conversation['lines'][i]['text'].strip()
            targetln = conversation['lines'][i + 1]['text'].strip()
            # Filter wrong samples / empty lists
            if inputln and targetln:
                qa_pairs.append([inputln, targetln])
    return qa_pairs


# Create new file & define path to it
datafile = os.path.join(corpus, 'formatted_movie_lines.txt')
# Set & unescape delimiter
delimiter = str(codecs.decode('\t', 'unicode_escape'))
# Init lines dictionary, conversations list, & field ids
lines = {}
convs = []
MOVIE_LINES_FIELDS = ['lineID', 'charID', 'movieID', 'char', 'text']
MOVIE_CONVERSATIONS_FIELDS = ['char1ID', 'char2ID', 'movieID', 'utteranceIDs']
# Load lines & process conversations
print('\nProcessing corpus...')
lines = loadlns(os.path.join(corpus, 'movie_lines.txt'), MOVIE_LINES_FIELDS)
print('\nLoading conversations...')
conversations = loadconv(os.path.join(corpus, 'movie_conversations.txt'), lines, MOVIE_CONVERSATIONS_FIELDS)
# Write new csv file
print('\nWriting newly formatted file...')
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for pair in extractsentencepairs(conversations):
        writer.writerow(pair)
timeelapsed = (time.perf_counter_ns() / 1e+6) - timestart
print(f'\nDone, took {timeelapsed} ms;')
print('\n\nSample lines from file:\n')
println(datafile)

# Load & trim data
# Word tokens
PADDING_token = 0
STARTOFSENTENCE_token = 1
ENDOFSENTENCE_token = 2


class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.wordtoindex = {}
        self.wordtocount = {}
        self.indextoword = {
            PADDING_token: "PAD",
            STARTOFSENTENCE_token: "SOS",
            ENDOFSENTENCE_token: "EOS"
        }
        self.numwords = 3

    def addsentence(self, sentence):
        for word in sentence.split(' '):
            self.addword(word)
    def addword(self, word):
        if word not in self.wordtoindex:
            self.wordtoindex[word] = self.numwords
            self.wordtocount[word] = 1
            self.indextoword[self.numwords] = word
            self.numwords += 1
        else:
            self.wordtocount[word] += 1
    # Trim words below count threshold
    def trim(self, min_threshold):
        if self.trimmed:
            return
        self.trimmed = True
        kept_words = []
        for keep, count in self.wordtocount.items():
            if count >= min_threshold:
                kept_words.append(keep)
        print('kept_words {} / {} = {:.4f}'.format(len(kept_words), len(self.wordtoindex), len(kept_words) / len(self.wordtoindex)))
        # Re-init dictionaries
        self.wordtoindex = {}
        self.wordtocount = {}
        self.indextoword = {
            PADDING_token: 'PAD',
            STARTOFSENTENCE_token: 'SOS',
            ENDOFSENTENCE_token: 'EOS'
        }
        self.numwords = 3
        for word in kept_words:
            self.addword(word)

# Preprocessing & normalization
# Turn unicode string to ASCII
def unicodetoascii(strg):
    return ''.join(
        char for char in unicodedata.normalize('NFD', strg) if unicodedata.category(char) is not 'Mn'
    )
# Normalize: lowercase, trim & remove non letter chars
def normalizestring(strg):
    strg = unicodetoascii(strg.lower().strip())
    strg = re.sub(r"([.!?])", r" \1", strg)
    strg = re.sub(r"[^a-zA-Z.!?]+", r" ", strg)

# endregion
