import math
import struct
import numpy as np
from multiprocessing import Pool, Value, Array

def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))

def init_net(dim, library_size):
    tmp = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(library_size, dim))
    weight0 = np.ctypeslib.as_ctypes(tmp)
    weight0 = Array(weight0._type_, weight0, lock=False)

    tmp = np.zeros(shape=(library_size, dim))
    weight1 = np.ctypeslib.as_ctypes(tmp)
    weight1 = Array(weight1._type_, weight1, lock=False)

    return (weight0, weight1)

class UnigramTable:
    def __init__(self, library):
        library_size = len(library)
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in library]) 

        table_size = 1e8 
        table = np.zeros(table_size, dtype=np.uint32)

        p = 0 # Cumulative probability
        i = 0
        for j, unigram in enumerate(library):
            p += float(math.pow(unigram.count, power))/norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]

class LibraryComponent:
    def __init__(self, word):
        self.word = word
        self.count = 0
        self.path = None 
        self.code = None

class Library:
    def __init__(self, fi, min_count):
        library_items = []
        library_hash = {}
        word_count = 0
        fi = open(fi, 'r')

        for token in ['<bol>', '<eol>']:
            library_hash[token] = len(library_items)
            library_items.append(LibraryComponent(token))

        for line in fi:
            tokens = line.split()
            for token in tokens:
                if token not in library_hash:
                    library_hash[token] = len(library_items)
                    library_items.append(LibraryComponent(token))
                    
                library_items[library_hash[token]].count += 1
                word_count += 1
            
            library_items[library_hash['<bol>']].count += 1
            library_items[library_hash['<eol>']].count += 1
            word_count += 2

        self.bytes = fi.tell()
        self.library_items = library_items         
        self.library_hash = library_hash     
        self.word_count = word_count
        self.__sort(min_count)
        print 'Vocabulary size: %d' % len(self)

    def __getitem__(self, i):
        return self.library_items[i]

    def __len__(self):
        return len(self.library_items)

    def __iter__(self):
        return iter(self.library_items)

    def __contains__(self, key):
        return key in self.library_hash

    def __sort(self, min_count):
        tmp = []
        tmp.append(LibraryComponent('<unk>'))
        unk_hash = 0
        
        count_unk = 0
        for token in self.library_items:
            if token.count < min_count:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token)

        tmp.sort(key=lambda token : token.count, reverse=True)

        # Update library_hash
        library_hash = {}
        for i, token in enumerate(tmp):
            library_hash[token.word] = i

        self.library_items = tmp
        self.library_hash = library_hash

    def indices(self, tokens):
        return [self.library_hash[token] if token in self else self.library_hash['<unk>'] for token in tokens]

    def encode_huffman(self):
        library_size = len(self)
        count = [t.count for t in self] + [1e15] * (library_size - 1)
        parent = [0] * (2 * library_size - 2)
        binary = [0] * (2 * library_size - 2)
        
        pos1 = library_size - 1
        pos2 = library_size

        for i in xrange(library_size - 1):
            #min1
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

            #min2
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

            count[library_size + i] = count[min1] + count[min2]
            parent[min1] = library_size + i
            parent[min2] = library_size + i
            binary[min2] = 1

        root_idx = 2 * library_size - 2
        for i, token in enumerate(self):
            path = [] 
            code = []
            
            node_idx = i
            while node_idx < root_idx:
                if node_idx >= library_size: path.append(node_idx)
                code.append(binary[node_idx])
                node_idx = parent[node_idx]
            path.append(root_idx)

            # These are path and code from the root to the leaf
            token.path = [j - library_size for j in path[::-1]]
            token.code = code[::-1]

def train_process(pid):
    start = library.bytes / num_processes * pid
    end = library.bytes if pid == num_processes - 1 else library.bytes / num_processes * (pid + 1)
    fi.seek(start)

    alpha = starting_alpha

    word_count = 0
    last_word_count = 0

    while fi.tell() < end:
        line = fi.readline().strip()
        if not line:
            continue
        sent = library.indices(['<bol>'] + line.split() + ['<eol>'])

        for sent_pos, token in enumerate(sent):
            if word_count % 10000 == 0:
                global_word_count.value += (word_count - last_word_count)
                last_word_count = word_count
                alpha = starting_alpha * (1 - float(global_word_count.value) / library.word_count)
                if alpha < starting_alpha * 0.0001: alpha = starting_alpha * 0.0001

            current_win = np.random.randint(low=1, high=win+1)
            context_start = max(sent_pos - current_win, 0)
            context_end = min(sent_pos + current_win + 1, len(sent))
            context = sent[context_start:sent_pos] + sent[sent_pos+1:context_end]

            neu1 = np.mean(np.array([weight0[c] for c in context]), axis=0)
            assert len(neu1) == dim, 'neu1 and dim do not agree'

            neu1e = np.zeros(dim)

            if neg > 0:
                classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
            else:
                classifiers = zip(library[token].path, library[token].code)
            for target, label in classifiers:
                z = np.dot(neu1, weight1[target])
                p = sigmoid(z)
                g = alpha * (label - p)
                neu1e += g * weight1[target]
                weight1[target] += g * neu1

            for context_word in context:
                weight0[context_word] += neu1e


            word_count += 1

    fi.close()

def save(library, weight0, fo, binary):
    print 'Saving model to', fo
    dim = len(weight0[0])
    if binary:
        fo = open(fo, 'wb')
        fo.write('%d %d\n' % (len(weight0), dim))
        fo.write('\n')
        for token, vector in zip(library, weight0):
            fo.write('%s ' % token.word)
            for s in vector:
                fo.write(struct.pack('f', s))
            fo.write('\n')
    else:
        fo = open(fo, 'w')
        fo.write('%d %d\n' % (len(weight0), dim))
        for token, vector in zip(library, weight0):
            word = token.word
            vector_str = ' '.join([str(s) for s in vector])
            fo.write('%s %s\n' % (word, vector_str))

    fo.close()

def __init_process(*args):
    global library, weight0, weight1, table, neg, dim, starting_alpha
    global win, num_processes, global_word_count, fi
    
    library, weight0_tmp, weight1_tmp, table, neg, dim, starting_alpha, win, num_processes, global_word_count = args[:-1]
    fi = open(args[-1], 'r')
    weight0 = np.ctypeslib.as_array(weight0_tmp)
    weight1 = np.ctypeslib.as_array(weight1_tmp)

def train(fi, fo, neg, dim, alpha, win, min_count, num_processes, binary):
    # Read train file to init library
    library = Library(fi, min_count)

    # Init net
    weight0, weight1 = init_net(dim, len(library))

    global_word_count = Value('i', 0)
    table = None
    if neg > 0:
        print 'Initializing unigram table'
        table = UnigramTable(library)
    else:
        print 'Initializing Huffman tree'
        library.encode_huffman()

    # Begin training using num_processes workers
    pool = Pool(processes=num_processes, initializer=__init_process,
                initargs=(library, weight0, weight1, table, neg, dim, alpha,
                          win, num_processes, global_word_count, fi))
    pool.map(train_process, range(num_processes))
    print
    print 'Completed training'

    # Save model to file
    save(library, weight0, fo, binary)

if __name__ == '__main__':
    train("datatrain", "dataout", 0, 300, 0.025, 5, 5, 1, 0)
