# -*- coding: utf-8 -*-
import os
import logging
from gensim.models import word2vec
import itertools
from glove import Corpus, Glove
import glove
import gensim
from gensim import utils
import smart_open

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("run_embed")

def listprinter(l):
    if isinstance(l, list):
        return u'[' + u','.join([listprinter(i[0]) for i in l]) + u']'
    elif isinstance(l, (str, unicode)):
        return u"'" + unicode(l) + u"'"
    elif isinstance(l, tuple):
        return u'(' + u','.join([listprinter(i[0]) for i in l]) + u')'


def glove2word2vec(glove_vector_file, output_model_file):
    """Convert GloVe vectors into word2vec C format"""

    def get_info(glove_file_name):
        """Return the number of vectors and dimensions in a file in GloVe format."""
        with smart_open.smart_open(glove_file_name) as f:
            num_lines = sum(1 for line in f)
        with smart_open.smart_open(glove_file_name) as f:
            num_dims = len(f.readline().split()) - 1
        return num_lines, num_dims
 
    def prepend_line(infile, outfile, line):
        """
        Function to prepend lines using smart_open
        """
        with smart_open.smart_open(infile, 'rb') as old:
            with smart_open.smart_open(outfile, 'wb') as new:
                new.write(str(line.strip()) + "\n")
                for line in old:
                    new.write(line)
        return outfile
 
    num_lines, dims = get_info(glove_vector_file)
 
    logger.info('%d lines with %s dimensions' % (num_lines, dims))
 
    gensim_first_line = "{} {}".format(num_lines, dims)
    model_file = prepend_line(glove_vector_file, output_model_file, gensim_first_line)
 
    logger.info('Model %s successfully created !!'%output_model_file)

    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False) #GloVe Model

    logger.info("Finished running")

    return model


#Load Corpus
myfile = 'joindataout/out_0_800.txt'
sentences = word2vec.LineSentence(myfile)

# outf = lambda prefix: os.path.join('output', prefix)
# if os.path.exists(outf('word2id')):
#     logger.info("dictionary found, loading")
#     word2id = utils.unpickle(outf('word2id'))
# else:
#     logger.info("dictionary not found, creating")
#     sentences_idword = lambda: itertools.islice(sentences, 10000000)
#     id2word = gensim.corpora.Dictionary(sentences_idword(), prune_at=10000000)
#     id2word.filter_extremes(keep_n=1000000)  # filter out too freq/infreq words
#     word2id = dict((v, k) for k, v in id2word.iteritems())
#     utils.pickle(word2id, outf('word2id'))

# print [x for x in sentences]
# print '--------------------------------------------------------------------'
# print 'TRAIN WORD VECTOR WITH SVD'
# print '--------------------------------------------------------------------'
# 
# #Train word vector with Skip Gram model
# try: #try to load if it's has trained before
#     SVDModel = SVDWord2Vec('datatrain')
#     print SVDModel.most_similar('man', topn=5)
# except IOError: #else, training
#     SVDModel = word2vec.Word2Vec(sentences, sg=1, hs=0, size=300, window=5, workers=4)
#     SVDModel.save('skipgramText8.model')
#     
# print SVDModel
# print '--------------------------------------------------------------------'
# print 'SVD FINISH'
# print '--------------------------------------------------------------------'

print '--------------------------------------------------------------------'
print 'TRAIN WORD VECTOR WITH SKIP-GRAM'
print '--------------------------------------------------------------------'

#Train word vector with Skip Gram model
try: #try to load if it's has trained before
    skipGramModel = word2vec.Word2Vec.load('output/skipgram800.model')
except IOError: #else, training
    skipGramModel = word2vec.Word2Vec(sentences, sg=1, hs=0, size=300, window=5, workers=8)
    skipGramModel.save('output/skipgram800.model')
     
print skipGramModel
print '--------------------------------------------------------------------'
print 'SKIP-GRAM FINISH'
print '--------------------------------------------------------------------'
 
print '--------------------------------------------------------------------'
print 'TRAIN WORD VECTOR WITH CBOW'
print '--------------------------------------------------------------------'
 
#Train word vector with CBOW model
try:
    CBOWModel = word2vec.Word2Vec.load('output/CBOW800.model')
except IOError:
    CBOWModel = word2vec.Word2Vec(sentences, sg=0, hs=1, size=300, window=5, workers=8) 
    CBOWModel.save('output/CBOW800.model')
     
print CBOWModel
print '--------------------------------------------------------------------'
print 'CBOW FINISH'
print '--------------------------------------------------------------------'
 
print '--------------------------------------------------------------------'
print 'TRAIN WORD VECTOR WITH GLOVE'
print '--------------------------------------------------------------------'
 
# #Train word vector with Glove
# try:
#     glove = Glove.load('output/GloveText8.model')
#     glove.word2id = dict((utils.to_unicode(w), id) for w, id in glove.dictionary.iteritems())
#     glove.id2word = gensim.utils.revdict(glove.word2id)
# except IOError:
#     glove_sentences = list(itertools.islice(word2vec.LineSentence(myfile), None))
#     corpus = Corpus()
#     corpus.fit(sentences)
#     glove = Glove(no_components=300, learning_rate=0.05)
#     glove.fit(corpus.matrix, epochs=100, no_threads=8, verbose=True)
#     glove.add_dictionary(corpus.dictionary)
#     glove.word2id = dict((utils.to_unicode(w), id) for w, id in glove.dictionary.iteritems())
#     glove.id2word = gensim.utils.revdict(glove.word2id)
#     glove.save('output/GloveText8.model')

glove = glove2word2vec('output/vectors_650.txt', 'output/gensim_Glove_vectors_650') 
  
print glove
print '--------------------------------------------------------------------'
print 'GLOVE FINISH'
print '--------------------------------------------------------------------'
    
    
# def benmarkSimilar(token, top):
#     skipModel = skipGramModel.most_similar([token], topn=top)
#     cbowModel = CBOWModel.most_similar([token], topn=top)
#     gloveModel = glove.most_similar(token, number=top)
#     print '--------------------------------------------------------------------'
#     print 'BENMARK SIMILAR : ', token
#     print 'Skip-Gram : ', listprinter(skipModel)
#     print 'CBOW : ', listprinter(cbowModel)
#     print 'Glove : ', listprinter(gloveModel)
#     print '--------------------------------------------------------------------'
#     
# def benmarkSenmatic(token1, token2, token3, top):
#     skipModel = skipGramModel.most_similar([token1, token2], [token3], topn=top)
#     cbowModel = CBOWModel.most_similar([token1, token2], [token3], topn=top)
# 
#     print '--------------------------------------------------------------------'
#     print 'BENMARK SENMATIC : ', token1, ' -> ', token2, ' -- ', token3
#     print 'Skip-Gram : ', listprinter(skipModel)
#     print 'CBOW : ', listprinter(cbowModel)
#     print '--------------------------------------------------------------------'

def benmarkAccuracy(filename):
    print '--------------------------------------------------------------------'
    skipGramModel.accuracy(filename)
    print '--------------------------------------------------------------------'
    print '--------------------------------------------------------------------'
    CBOWModel.accuracy(filename)
    print '--------------------------------------------------------------------'
    print '--------------------------------------------------------------------'
    glove.accuracy(filename)
    print '--------------------------------------------------------------------'

# benmarkSimilar(u'Ông', top=5)
# benmarkSimilar(u'Cháu', top=5)
# benmarkSimilar(u'nhanh', top=5)
# benmarkSimilar(u'nóng', top=5)
# benmarkSimilar(u'lạnh', top=5)
# benmarkSimilar(u'thích', top=5)
# benmarkSimilar(u'chạy', top=5)
# benmarkSimilar(u'nhảy', top=5)
# benmarkSimilar(u'chậm_hơn', top=5)

benmarkAccuracy('question');

