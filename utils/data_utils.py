# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 15:53:20 2016

@author: duytinvo
"""
from collections import Counter
import numpy,csv,sys

class Pipe:
    @staticmethod
    def process_line(x,lowercase=True):
#        line=line.replace('\\n', ' \n ')
#        line=line.replace('\\""', ' \" ')
        if lowercase:
            x=x.lower()
        x = x.strip().split()
        return x
        
    @staticmethod
    def csvfile(rfile,firstline=True): 
        maxInt = sys.maxsize
        decrement = True
        while decrement:
            # decrease the maxInt value by factor 10 
            # as long as the OverflowError occurs.
            decrement = False
            try:
                csv.field_size_limit(maxInt)
            except OverflowError:
                maxInt = int(maxInt/10)
                decrement = True
        with open(rfile, 'rb') as f:
            rows = csv.reader(f)
            for row in rows:
                if firstline:
                    firstline=False
                    continue
#                line=' '.join(row)
                x,y=row
                x= Pipe.process_line(x)
                yield x,y 
        
    @staticmethod
    def create_vocabulary(train_file,cutoff):
        cnty = Counter()
        cntx = Counter()
        cntc = Counter()
        print "Build vocabulary..."
        raw=Pipe.csvfile(train_file)  
        lw=0
        lc=0
        for xy in raw:
            cntx.update(xy[0])
            lw=max(lw,len(xy[0]))
            cntc.update(' '.join(xy[0]))
            lc=max(lc,max([len(wd) for wd in xy[0]]))
            cnty.update([xy[1]])
        print  "%d total words, %d total characters, %d total labels" % (len(cntx),len(cntc),len(cnty))
        Vocabx=[x for x, y in cntx.iteritems() if y >= cutoff]
        lstx = [u"<PADw>"] + Vocabx + [u"<UNKw>"]
        vocabx = dict([ (y,x) for x,y in enumerate(lstx) ])
        Vocabc=[x for x, y in cntc.iteritems() if y >= cutoff]
        lstc = [u"<PADc>"] + Vocabc + [u"<UNKc>"]
        vocabc = dict([ (y,x) for x,y in enumerate(lstc) ])
        vocaby = dict([ (y,x) for x,y in enumerate(cnty.keys()) ])
        print  "%d unique words, %d unique characters appearing at least %d times" % (len(vocabx)-2,len(vocabc)-2, cutoff)
        return vocabx, vocabc, vocaby, lw, lc

    @staticmethod
    def ch2idx(x, vocabc,lw,lc,padc):
        c = []
        for w in x[:lw]:
            cid=[vocabc['<PADc>']]*padc + [vocabc.get(ch,vocabc['<UNKc>']) for ch in w[:lc]]
            if len(cid)<lc+2*padc:
                cid=cid+[vocabc['<PADc>']]*(lc+2*padc-len(cid))
            c.extend(cid)
        if len(c) < lw*(lc+2*padc):
            c=c+[vocabc['<PADc>']]*(lw*(lc+2*padc)-len(c))
        return c
    
    @staticmethod
    def wd2idx(x, vocabx, lw, padw):
        xid=[vocabx['<PADw>']]*padw + [vocabx.get(wd,vocabx['<UNKw>']) for wd in x[:lw]]
        if len(xid) < (lw+2*padw):
            xid=xid+[vocabx['<PADw>']]*((lw+2*padw)-len(xid))
        return xid

    @staticmethod
    def sent2seq(rfile, vocabx, vocabc, vocaby, lw_th=1024, lc_th=32, padw=0, padc=0):
        corpus_x = [ ]
        corpus_c = [ ]
        corpus_y = [ ]
        oov_x= []
        oov_c= []
        raw =Pipe.csvfile(rfile)
        print("Convert text to sequence...")
        for xy in raw:  
            oov_x.extend([1.0 if w not in vocabx else 0 for w in xy[0]])
            xid=Pipe.wd2idx(xy[0], vocabx,lw_th,padw)
            corpus_x.append(xid)
            oov_c.extend([1.0 if ch not in vocabc else 0 for w in xy[0] for ch in w])
            cid=Pipe.ch2idx(xy[0], vocabc,lw_th,lc_th,padc)
            corpus_c.append(cid)
            y=vocaby[xy[1]]
            corpus_y.append(y)
        print "{}: size={}, word_oov rate={}".format(rfile, len(corpus_x), sum(oov_x)/float(len(oov_x)))
        corpus_c=numpy.array(corpus_c,dtype="int16")
        corpus_x=numpy.array(corpus_x,dtype="int32")
        corpus_y=numpy.array(corpus_y,dtype="int16")
        return corpus_x, corpus_c, corpus_y

    @staticmethod
    def to_categorical(y, nb_classes=None):
        '''Convert class vector (integers from 0 to nb_classes)
        to binary class matrix, for use with categorical_crossentropy.
        '''
        if not nb_classes:
            nb_classes = numpy.max(y)+1
        Y = numpy.zeros((len(y), nb_classes),dtype="int16")
        for i in range(len(y)):
            Y[i, y[i]] = 1
        return Y 

    @staticmethod
    def load_embs(fname):
        embs=dict()
        s=0
        V=0
        with open(fname,'rb') as f:
            for line in f: 
                p=line.strip().split()
                if len(p)==2:
                    V=int(p[0])
                    s=int(p[1])
                else:
#                    assert len(p)== s+1
                    w=p[0]
                    e=[float(i) for i in p[1:]]
                    embs[w]=numpy.array(e,dtype="float32")
#        assert len(embs)==V
        return embs 
    
    @staticmethod
    def get_W(emb_file, wsize, vocabx, scale=0.25):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        word_vecs =Pipe.load_embs(emb_file)
        word_vecs[u'<PAD>']=numpy.zeros(shape=(1,wsize),dtype="float32")
        
        unk=0
        W = numpy.zeros(shape=(len(vocabx), wsize),dtype="float32")            
        for word,idx in vocabx.iteritems():
            vector = word_vecs.get(word)
            if vector is not None:
                W[idx]=vector
            else:
                unk+=1
                rvector=numpy.asarray(numpy.random.uniform(-scale,scale,(1,wsize)),dtype="float32")
                W[idx]=rvector
        print '\t', unk, 'words not in pre-trained vectors are initialized randomly;'
        print '\t', len(vocabx)-unk, 'pre-trained embeddings.'
#        W[1:]=W[1:]/numpy.sqrt((W[1:]**2).sum(axis=1,keepdims=True))
        return W
            
if __name__ == "__main__":
    trfile='../../metadata/datasets/train.csv'
    tfile='../../metadata/datasets/test.csv'
    vocabx, vocabc, vocaby, lw, lc = Pipe.create_vocabulary(trfile,1)
    x_train_wd, x_train_ch, y_train = Pipe.sent2seq(trfile, vocabx, vocabc, vocaby, lw, lc)
