#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Jun 24, 2017

@author: CuongNguyen
'''
import argparse
import math
import struct
import sys
import time
import warnings
import re
import numpy as np

from multiprocessing import Pool, Value, Array
import copy

class VocabItem:
    def __init__(self, word):
        self.word = word
        self.count = 0
        self.path = None # Path (list of indices) from the root to the word (leaf)
        self.code = None # Huffman encoding

class Vocab:
    def __init__(self, fi, min_count):
        
        with open("bigram_table") as f:
            content = f.readlines()
        self.bigram_items = [x.strip() for x in content] 

        vocab_items = []
        vocab_hash = {}
        word_count = 0
        fi = open(fi, 'r')

        # Add special tokens <bol> (beginning of line) and <eol> (end of line)
        for token in ['<bol>', '<eol>']:
            vocab_hash[token] = len(vocab_items)
            vocab_items.append(VocabItem(token))

        for line in fi:
            tokens = line.split()
            for token in tokens:
                if token not in vocab_hash:
                    vocab_hash[token] = len(vocab_items)
                    vocab_items.append(VocabItem(token))
                    
                #assert vocab_items[vocab_hash[token]].word == token, 'Wrong vocab_hash index'
                vocab_items[vocab_hash[token]].count += 1
                word_count += 1
            
                if word_count % 100000 == 0:
                    sys.stdout.write("\rReading word %d" % word_count)
                    sys.stdout.flush()

            # Add special tokens <bol> (beginning of line) and <eol> (end of line)
            vocab_items[vocab_hash['<bol>']].count += 1
            vocab_items[vocab_hash['<eol>']].count += 1
            word_count += 2

        
        sys.stdout.write("\rReading bigram word %d" % len(self.bigram_items))

        for token in self.bigram_items:
            if token not in vocab_hash:
                vocab_hash[token] = len(vocab_items)
                vocab_items.append(VocabItem(token))
            vocab_items[vocab_hash[token]].count += 10
            word_count += 1
        
        self.bytes = fi.tell()
        self.vocab_items = vocab_items         # List of VocabItem objects
        self.vocab_hash = vocab_hash           # Mapping from each token to its index in vocab
        self.word_count = word_count           # Total number of words in train file

        # Add special token <unk> (unknown),
        # merge words occurring less than min_count into <unk>, and
        # sort vocab in descending order by frequency in train file
        self.__sort(min_count)

        #assert self.word_count == sum([t.count for t in self.vocab_items]), 'word_count and sum of t.count do not agree'
        print 'Total words in training file: %d' % self.word_count
        print 'Total bytes in training file: %d' % self.bytes
        print 'Vocab size: %d' % len(self)

    def __getitem__(self, i):
        return self.vocab_items[i]

    def __len__(self):
        return len(self.vocab_items)

    def __iter__(self):
        return iter(self.vocab_items)

    def __contains__(self, key):
        return key in self.vocab_hash

    def __sort(self, min_count):
        tmp = []
        tmp.append(VocabItem('<unk>'))
        unk_hash = 0
        
        count_unk = 0
        for token in self.vocab_items:
            if token.count < min_count:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token)

        tmp.sort(key=lambda token : token.count, reverse=True)

        # Update vocab_hash
        vocab_hash = {}
        for i, token in enumerate(tmp):
            vocab_hash[token.word] = i

        self.vocab_items = tmp
        self.vocab_hash = vocab_hash

        print
        print 'Unknown vocab size:', count_unk

    def indices(self, tokens):
        return [self.vocab_hash[token] if token in self else self.vocab_hash['<unk>'] for token in tokens]

    def encode_huffman(self):
        # Build a Huffman tree
        vocab_size = len(self)
        count = [t.count for t in self] + [1e15] * (vocab_size - 1)
        parent = [0] * (2 * vocab_size - 2)
        binary = [0] * (2 * vocab_size - 2)
        
        pos1 = vocab_size - 1
        pos2 = vocab_size

        for i in xrange(vocab_size - 1):
            # Find min1
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min1 = pos1
                    pos1 -= 1
                else:
                    min1 = pos2
                    pos2 += 1
            else:
                min1 = pos2
                pos2 += 1

            # Find min2
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min2 = pos1
                    pos1 -= 1
                else:
                    min2 = pos2
                    pos2 += 1
            else:
                min2 = pos2
                pos2 += 1

            count[vocab_size + i] = count[min1] + count[min2]
            parent[min1] = vocab_size + i
            parent[min2] = vocab_size + i
            binary[min2] = 1

        # Assign binary code and path pointers to each vocab word
        root_idx = 2 * vocab_size - 2
        for i, token in enumerate(self):
            path = [] # List of indices from the leaf to the root
            code = [] # Binary Huffman encoding from the leaf to the root

            node_idx = i
            while node_idx < root_idx:
                if node_idx >= vocab_size: path.append(node_idx)
                code.append(binary[node_idx])
                node_idx = parent[node_idx]
            path.append(root_idx)

            # These are path and code from the root to the leaf
            token.path = [j - vocab_size for j in path[::-1]]
            token.code = code[::-1]

class UnigramTable:
    """
    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """
    def __init__(self, vocab):
        vocab_size = len(vocab)
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab]) # Normalizing constant

#         table_size = 100000000 # Length of the unigram table
        table_size = 10000000 # Length of the unigram table

        table = np.zeros(table_size, dtype=np.uint32)

        print 'Filling unigram table'
        p = 0 # Cumulative probability
        i = 0
        for j, unigram in enumerate(vocab):
            p += float(math.pow(unigram.count, power))/norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]

def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))

def init_net(dim, vocab_size):
    # Init syn0 with random numbers from a uniform distribution on the interval [-0.5, 0.5]/dim
    tmp = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(vocab_size, dim))
    syn0 = np.ctypeslib.as_ctypes(tmp)
    syn0 = Array(syn0._type_, syn0, lock=False)

    # Init syn1 with zeros
    tmp = np.zeros(shape=(vocab_size, dim))
    syn1 = np.ctypeslib.as_ctypes(tmp)
    syn1 = Array(syn1._type_, syn1, lock=False)

    return (syn0, syn1)

def train_process(pid):
    # Set fi to point to the right chunk of training file
    start = vocab.bytes / num_processes * pid
    end = vocab.bytes if pid == num_processes - 1 else vocab.bytes / num_processes * (pid + 1)
    fi.seek(start)
    #print 'Worker %d beginning training at %d, ending at %d' % (pid, start, end)

    alpha = starting_alpha

    word_count = 0
    last_word_count = 0

    while fi.tell() < end:
        line = fi.readline().strip()
        # Skip blank lines
        if not line:
            continue

        # Init sent, a list of indices of words in line
        sent = vocab.indices(['<bol>'] + line.split() + ['<eol>'])

        for sent_pos, token in enumerate(sent):
            if word_count % 10000 == 0:
                global_word_count.value += (word_count - last_word_count)
                last_word_count = word_count

                # Recalculate alpha
                alpha = starting_alpha * (1 - float(global_word_count.value) / vocab.word_count)
                if alpha < starting_alpha * 0.0001: alpha = starting_alpha * 0.0001

                # Print progress info
                sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                                 (alpha, global_word_count.value, vocab.word_count,
                                  float(global_word_count.value) / vocab.word_count * 100))
                sys.stdout.flush()            
            
            # Randomize window size, where win is the max window size
            current_win = np.random.randint(low=1, high=win+1)
            context_start = max(sent_pos - current_win, 0)
            context_end = min(sent_pos + current_win + 1, len(sent))
            context = sent[context_start:sent_pos] + sent[sent_pos+1:context_end] # Turn into an iterator?
            
            #bigram model
            for index,word in enumerate(context) :
                if index + 1 < len(context):
                    word_bi_gram = vocab.vocab_items[context[index]].word + "_" + vocab.vocab_items[context[index+1]].word
                    if word_bi_gram in vocab.bigram_items and vocab.vocab_hash[word_bi_gram] != None:
                        context.append(vocab.vocab_hash[word_bi_gram])
            
            for context_word in context:
                # Init neu1e with zeros
                neu1e = np.zeros(dim)

                # Compute neu1e and update syn1
                if neg > 0:
                    classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                else:
                    classifiers = zip(vocab[token].path, vocab[token].code)
                for target, label in classifiers:
                    z = np.dot(syn0[context_word], syn1[target])
                    p = sigmoid(z)
                    g = alpha * (label - p)
                    neu1e += g * syn1[target]              # Error to backpropagate to syn0
                    syn1[target] += g * syn0[context_word] # Update syn1

                # Update syn0
                syn0[context_word] += neu1e

             word_count += 1
            
             #biram model
             if sent_pos + 1 < len(sent):
                 token_next = vocab.vocab_items[sent[sent_pos]].word + "_" + vocab.vocab_items[sent[sent_pos+1]].word
                 check = token_next in vocab.bigram_items
                 if check and vocab.vocab_hash[token_next] != None:
                     sent_copy = copy.copy(sent)
                     sent_copy[sent_pos] = vocab.vocab_hash[token_next]
                     sent_copy.pop(sent_pos+1)
                     
                     # Randomize window size, where win is the max window size
                     current_win = np.random.randint(low=1, high=win+1)
                     context_start = max(sent_pos - current_win, 0)
                     context_end = min(sent_pos + current_win + 1, len(sent_copy))
                     context = sent_copy[context_start:sent_pos] + sent_copy[sent_pos+1:context_end]
                     
                     #bigram model
                     for index,word in enumerate(context) :
                         if index + 1 < len(context):
                             word_bi_gram = vocab.vocab_items[context[index]].word + "_" + vocab.vocab_items[context[index+1]].word
                             if word_bi_gram in vocab.bigram_items and vocab.vocab_hash[word_bi_gram] != None:
                                 context.append(vocab.vocab_hash[word_bi_gram])
                       
                     for context_word in context:
                         # Init neu1e with zeros
                         neu1e = np.zeros(dim)
       
                         # Compute neu1e and update syn1
                         if neg > 0:
                             classifiers = [(vocab.vocab_hash[token_next], 1)] + [(target, 0) for target in table.sample(neg)]
                         else:
                             classifiers = zip(vocab[token].path, vocab[token].code)
                         for target, label in classifiers:
                             z = np.dot(syn0[context_word], syn1[target])
                             p = sigmoid(z)
                             g = alpha * (label - p)
                             neu1e += g * syn1[target]              # Error to backpropagate to syn0
                             syn1[target] += g * syn0[context_word] # Update syn1

                         # Update syn0
                         syn0[context_word] += neu1e
        

    # Print progress info
    global_word_count.value += (word_count - last_word_count)
    sys.stdout.write("DONE - \rAlpha: %f Progress: %d of %d (%.2f%%)" %
                     (alpha, global_word_count.value, vocab.word_count,
                      float(global_word_count.value)/vocab.word_count * 100))
    sys.stdout.flush()
    fi.close()

def save(vocab, syn0, fo, binary):
    print 'Saving model to', fo
    dim = len(syn0[0])
    if binary:
        fo = open(fo, 'wb')
        fo.write('%d %d\n' % (len(syn0), dim))
        fo.write('\n')
        for token, vector in zip(vocab, syn0):
            fo.write('%s ' % token.word)
            for s in vector:
                fo.write(struct.pack('f', s))
            fo.write('\n')
    else:
        fo = open(fo, 'w')
        fo.write('%d %d\n' % (len(syn0), dim))
        for token, vector in zip(vocab, syn0):
            word = token.word
            vector_str = ' '.join([str(s) for s in vector])
            fo.write('%s %s\n' % (word, vector_str))

    fo.close()

def __init_process(*args):
    global vocab, syn0, syn1, table, cbow, neg, dim, starting_alpha
    global win, num_processes, global_word_count, fi
    
    vocab, syn0_tmp, syn1_tmp, table, cbow, neg, dim, starting_alpha, win, num_processes, global_word_count = args[:-1]
    fi = open(args[-1], 'r')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        syn0 = np.ctypeslib.as_array(syn0_tmp)
        syn1 = np.ctypeslib.as_array(syn1_tmp)

def train(fi, fo, cbow, neg, dim, alpha, win, min_count, num_processes, binary):
    # Read train file to init vocab
    vocab = Vocab(fi, min_count)

    # Init net
    syn0, syn1 = init_net(dim, len(vocab))

    global_word_count = Value('i', 0)
    table = None
    if neg > 0:
        print 'Initializing unigram table'
        table = UnigramTable(vocab)
    else:
        print 'Initializing Huffman tree'
        vocab.encode_huffman()

    # Begin training using num_processes workers
    t0 = time.time()
    pool = Pool(processes=num_processes, initializer=__init_process,
                initargs=(vocab, syn0, syn1, table, cbow, neg, dim, alpha,
                          win, num_processes, global_word_count, fi))
    pool.map(train_process, range(num_processes))
    t1 = time.time()
    print
    print 'Completed training. Training took', (t1 - t0) / 60, 'minutes'

    # Save model to file
    save(vocab, syn0, fo, binary)

if __name__ == '__main__':
    train("100mb.txt", "half_vec_100mb.txt", 
          cbow=0, 
          neg=1, 
          dim=300, 
          alpha=0.025, 
          win=5, 
          min_count=5, 
          num_processes=8, 
          binary=0)
