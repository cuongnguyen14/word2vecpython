# -*- coding: utf-8 -*-
import logging, os
from gensim.models import word2vec
import gensim
import smart_open
import argparse
import shutil, glob
from random import shuffle
import sys
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy
import scipy.sparse
from gensim import utils, matutils
import numpy.core.numerictypes
import codecs

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("run_embed")

def visuallizer(model):
#     vocab = [u'chạy', u'đang_chạy'
# ,u'bơi', u'đang_bơi'
# ,u'học', u'đang_học'
# ,u'làm', u'đang_làm'
# ,u'đọc', u'đang_đọc'
# ,u'nói', u'đang_nói'
# ,u'nghe', u'đang_nghe'
# ,u'nhìn', u'đang_nhìn'
# ,u'chơi', u'đang_chơi'
# ,u'ngủ', u'đang_ngủ'
# ,u'thức', u'đang_thức']
    
#     vocab = [
#         u'to', u'to_hơn'
# ,u'lớn', u'lớn_hơn'
# ,u'nhỏ', u'nhỏ_hơn'
# ,u'nhanh', u'nhanh_hơn'
# ,u'chậm', u'chậm_hơn'
# ,u'lạnh', u'lạnh_hơn'
# ,u'tốt', u'tốt_hơn'
# ,u'xấu', u'xấu_hơn'
# ,u'trẻ', u'trẻ_hơn'
# ,u'già', u'già_hơn'
# ,u'rẻ', u'rẻ_hơn'
# ,u'mắc', u'mắc_hơn'
# ,u'đẹp', u'đẹp_hơn'
# ,u'dễ', u'dễ_hơn'
# ,u'sáng', u'sáng_hơn'
# ,u'mạnh', u'mạnh_hơn'
# ,u'rộng', u'rộng_hơn'
# ,u'giỏi', u'giỏi_hơn'
# ,u'nóng', u'nóng_hơn'
# ,u'tối', u'tối_hơn'
# ]

    vocab = [
        u'to', u'to_nhất'
    ,u'lớn', u'lớn_nhất'
    ,u'nhỏ', u'nhỏ_nhất'
    ,u'nhanh', u'nhanh_nhất'
    ,u'chậm', u'chậm_nhất'
    ,u'lạnh', u'lạnh_nhất'
    ,u'tốt', u'tốt_nhất'
    ,u'xấu', u'xấu_nhất'
    ,u'trẻ', u'trẻ_nhất'
    ,u'già', u'già_nhất'
    ,u'rẻ', u'rẻ_nhất'
    ,u'mắc', u'mắc_nhất'
    ,u'đẹp', u'đẹp_nhất'
    ,u'dễ', u'dễ_nhất'
    ,u'sáng', u'sáng_nhất'
    ,u'mạnh', u'mạnh_nhất'
    ,u'rộng', u'rộng_nhất'
    ,u'giỏi', u'giỏi_nhất'
    ,u'nóng', u'nóng_nhất'
    ,u'tối', u'tối_nhất'
    ]
#     vocab_ = list(model.wv.vocab)
#     vocab = vocab_[:30]
    X = model[vocab]
    tsne = TSNE(n_components=2)
    X_tsne1 = tsne.fit_transform(X)
    X_tsne = X_tsne1[:15]
    df = pd.concat([pd.DataFrame(X_tsne),
                    pd.Series(vocab)],
                    axis=1)

    df.columns = ['x', 'y', 'word']
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(df['x'], df['y'])
    for i, txt in enumerate(df['word']):
        ax.annotate(txt, (df['x'].iloc[i], df['y'].iloc[i]))    
    plt.show()

def listprinter(l):
    if isinstance(l, list):
        return u'[' + u','.join([listprinter(i[0]) for i in l]) + u']'
    elif isinstance(l, (str, unicode)):
        return u"'" + unicode(l) + u"'"
    elif isinstance(l, tuple):
        return u'(' + u','.join([listprinter(i[0]) for i in l]) + u')'

def listprinter2(l):
    return u'[' + u','.join([str(i[1]) for i in l]) + u']'
    
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


def generate_traning_data(input_folder, output_folder):
    in_file_name = output_folder + '/' + 'input_data.txt'

    with open(in_file_name, 'wb') as outfile:
        all_file = glob.glob1(input_folder,'*.txt')
        shuffle(all_file)
        for filename in all_file:
            with open(input_folder + '/' + filename, 'rb') as readfile:
                outfile.write('\n')
                shutil.copyfileobj(readfile, outfile)

    sentences = word2vec.LineSentence(in_file_name)

    return in_file_name, sentences

def train(method, input_folder, output_folder, dim=300, question='question', visual=False):
   
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    filename, sentences = generate_traning_data(input_folder, output_folder);
    
    if (method == 'skipgram'):
        print '--------------------------------------------------------------------'
        print 'TRAIN WORD VECTOR WITH SKIP-GRAM'
        print '--------------------------------------------------------------------'
        try: #try to load if it's has trained before
            skipGramModel = word2vec.Word2Vec.load(output_folder + '/skipgram_' + str(dim) + '.model')
        except IOError: #else, training
            skipGramModel = word2vec.Word2Vec(sentences, sg=1, hs=0, size=dim, window=5, workers=8)
            skipGramModel.save(output_folder + '/skipgram_' + str(dim) + '.model')
            skipGramModel.wv.save_word2vec_format(output_folder + '/w2v_format_skipgram_' + str(dim) + '.model')

        skipGramModel.accuracy(question)
        if visual : 
            visuallizer(skipGramModel)
        print '--------------------------------------------------------------------'

    if (method == 'cbow'):
        print '--------------------------------------------------------------------'
        print 'TRAIN WORD VECTOR WITH CBOW'
        print '--------------------------------------------------------------------'
        #Train word vector with CBOW model
        try:
            CBOWModel = word2vec.Word2Vec.load(output_folder + '/cbow_' + str(dim) + '.model')
        except IOError:
            CBOWModel = word2vec.Word2Vec(sentences, sg=0, hs=1, size=dim, window=5, workers=8) 
            CBOWModel.save(output_folder + '/cbow_' + str(dim) + '.model')
        CBOWModel.accuracy(question)
        if visual : 
            visuallizer(CBOWModel)
        print '--------------------------------------------------------------------'

def add2gram_by_average(filein, bigramvectorin, bigram):
#     model = gensim.models.KeyedVectors.load_word2vec_format(filein, binary=False)
    
    #load all 2gramword vector
    print 'read all 2 gram vector model'
    all_2gram_w2v = {}
    all_word = []
    vc_size = 0
    vt_size = 0
    with utils.smart_open(bigramvectorin) as fbigramin:
        header = utils.to_unicode(fbigramin.readline(), encoding='utf8')
        vc_size, vt_size = map(int, header.split())
        for line_no in xrange(vc_size):
            line = fbigramin.readline()
            parts = utils.to_unicode(line.rstrip(), encoding='utf8', errors='strict').split(" ")
            word, weights = parts[0], numpy.array(parts[1:])
            all_word.append(word)
            all_2gram_w2v[word] = weights.astype(numpy.float)

    print 'get all origin vector model'
    all_w2v = {}
    with utils.smart_open(filein) as fin:
        header = utils.to_unicode(fin.readline(), encoding='utf8')
        vocab_size, vector_size = map(int, header.split())
        for line_no in xrange(vocab_size):
            line = fin.readline()
            parts = utils.to_unicode(line.rstrip(), encoding='utf8', errors='strict').split(" ")
            word, weights = parts[0], numpy.array(parts[1:])
            all_w2v[word] = weights.astype(numpy.float)

    with open(bigram) as f2:
        line = f2.readlines()
    bigram_items = [x.strip() for x in line]
    
    total = ''
    count = 0
    
    print 'average 2 gram word'
    for bigram in bigram_items :
        if bigram.startswith(':') :
            print bigram
            continue
        split = utils.to_unicode(bigram.rstrip(), encoding='utf8', errors='strict').split(" ")
        for word in split:
            temp = utils.to_unicode(word.rstrip(), encoding='utf8', errors='strict').split("_")
            if len(temp) < 2 :
                all_2gram_w2v[word] = all_w2v[word]
                print 'update for : ' + word
                continue
            
            w1, w2 = temp[0], temp[1]
            if all_w2v.has_key(w1) and all_w2v.has_key(w2) : 
                wv1, wv2 = all_w2v.get(w1), all_w2v.get(w2)
                wv12 = (numpy.array(wv1) + numpy.array(wv2)) / 2.0
                all_2gram_w2v[word] = wv12
                count += 1
                print 'calculate for : ' + word
                
    print 'write to file'
    filename = 'neww2vec.model'
    text_file = codecs.open(filename, "w", 'utf8')
    text_file.write(str(vc_size) + " " + str(vt_size))
    for word in all_word :
        lineappend = word + " " + ' '.join([str(s) for s in all_2gram_w2v[word]])
        text_file.write("\n" + lineappend)
    text_file.close()
    print 'end'
    return filename

def list_topn_similarity(filein, bigram):
    
    model = gensim.models.KeyedVectors.load_word2vec_format(filein, binary=False)
    
    #load all 2gramword vector
    vocab_items = {}
    with open(bigram) as f:
        content = f.readlines()
    bigram_items = [x.strip() for x in content] 

    for bigram in bigram_items :
        if bigram.startswith(':') :
            continue
        split = utils.to_unicode(bigram.rstrip(), encoding='utf8', errors='strict').split(" ")
        for word in split:
            vocab_items[word] = 1;  

    for key in vocab_items.keys() :
        vocab_items[key] = model.similar_by_word(key, topn=5)
    
    line = ''
    for key in vocab_items.keys() :
        line = line + key + ',' + listprinter(vocab_items[key]) + '\n'
        line = line + key + ',' + listprinter2(vocab_items[key]) + '\n'

    print 'write to file'
    filename = filein + 'top5_similar.txt'
    text_file = codecs.open(filename, "w", 'utf8')
    text_file.write(line)
    text_file.close()
    print 'end'
    
if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument('-method', help='svd, skipgram, cbow, glove', dest='method', required=True)
#     parser.add_argument('-input', help='input folder', dest='input', required=True)
#     parser.add_argument('-output', help='output folder', dest='output', required=True)
#     parser.add_argument('-dim', help='dimmension of word vector', dest='dim', default=300, type=int)
#     parser.add_argument('-question', help='question word for validate', dest='question', required=True)
#     args = parser.parse_args()
#     train(args.method, args.input, args.output, args.dim, args.question)
    
# python word2vec.py -method skipgram -input dataout -output myout -dim 300 -question question



#     train('skipgram', '2graminput', '2gramoutput', 300, 'question')
#     train('skipgram', 'rawinput', 'rawoutput', 300, 'question')
#     newmodel = add2gram_by_average('rawoutput/w2v_format_skipgram_300.model','2gramoutput/w2v_format_skipgram_300.model', 'question')
   
#     model = gensim.models.KeyedVectors.load_word2vec_format('neww2vec.model', binary=False)
#     model.accuracy('question')
#     print model.similar_by_word('nhanh', topn=5)
# 
#     nmodel = gensim.models.KeyedVectors.load_word2vec_format('2gramoutput/w2v_format_skipgram_300.model', binary=False)
#     nmodel.accuracy('question')
#     print nmodel.similar_by_word('nhanh', topn=5)
    
    list_topn_similarity("neww2vec.model", "question")
    list_topn_similarity("2gramoutput/w2v_format_skipgram_300.model", "question")

    print 'END'
    
