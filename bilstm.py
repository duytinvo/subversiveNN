# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 11:34:24 2016

@author: duytinvo
"""
from keras.callbacks import EarlyStopping
import sys, argparse, time, numpy
from utils.core_nns import build_BiRNN,saveload
from utils.data_utils import Pipe  
import os    
 
class baseNNs:
    def __init__(self, args=None):
        self.args = args
    
    def buildModel(self,embeddings):
        wdmax_features=len(self.args.vocabx)
        wdims=self.args.wsize
        trainable=self.args.trainable
        
        cmax_features=len(self.args.vocabc)        
        cdims=self.args.csize
        cwdims=self.args.cwsize
        
        maxlw=self.args.maxlw
        maxlc=self.args.maxlc
        dropout_in=self.args.dropout_in
        dropout_out=self.args.dropout_out
        rnndims = self.args.rnndims
        maxlen=maxlc*maxlw
        
        n_out=self.args.n_out
        learning=self.args.learning 
        
        print('Build model...')
        model=build_BiRNN(embeddings,wdmax_features, wdims,trainable,cmax_features,
                           cdims,cwdims,maxlen,dropout_in,dropout_out, rnndims,n_out,maxlc)
               
        print('Compile model...')
        if n_out>2:            
            model.compile(loss='categorical_crossentropy',optimizer=learning,metrics=['accuracy'])
        else:
            model.compile(loss='binary_crossentropy',optimizer=learning,metrics=['accuracy'])
        return model
                    
    def training(self, model):
        x_train, c_train,y_train = Pipe.sent2seq(self.args.train_file, self.args.vocabx, self.args.vocabc, self.args.vocaby, self.args.maxlw, self.args.maxlc, 0,0)
        x_text, c_test, y_test = Pipe.sent2seq(self.args.test_file, self.args.vocabx, self.args.vocabc, self.args.vocaby, self.args.maxlw, self.args.maxlc, 0,0)        
        early_stopping = EarlyStopping(monitor='val_acc', patience=self.args.patience)
        if self.args.n_out>2:
            y_train = Pipe.to_categorical(y_train, self.args.n_out)
            y_test = Pipe.to_categorical(y_test, self.args.n_out)
        else:
            y_train=numpy.array(y_train)
            y_test=numpy.array(y_test)

        print('Train model...')
        model.fit([c_train,x_train], y_train,
                  batch_size=self.args.batch_size,
                  epochs=self.args.max_epochs,
                  validation_split=0.15,
                  callbacks=[early_stopping])
        saveload.save(model,self.args)
        try:
            score, acc = model.evaluate([c_test,x_text], y_test, batch_size=self.args.batch_size)
            print "\n\tValuating on testing dataset:"
            print '\t\tTest score: %f'% (score)
            print '\t\tTest accuracy: %.2f'% (100*acc)
            print '\t\tTest error: %.2f'% (100*(1-acc))
            with open(os.path.join(self.args.subfolder,'logging'),'wb') as f:
                f.write("\n\tValuating on testing dataset:")
                f.write("\n\t\tTest score: %f"% (score))
                f.write("\n\t\tTest accuracy: %.2f"% (100*acc))
                f.write("\n\t\tTest error: %.2f\n"% (100*(1-acc)))
        except:
            print "Out of memory"
            pass
                    
def main(args):
    folder = os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder): 
        os.mkdir(folder)
    args.folder=folder
    subfolder = os.path.join(args.folder,os.path.basename(os.path.dirname(args.train_file)))
    if not os.path.exists(subfolder): 
        os.mkdir(subfolder)
    args.subfolder=subfolder
    
    print args
    if args.seed >= 0:
        seed = args.seed
    else:
        seed = int(time.time()*1000) % 9999
    print "seed:", seed
    numpy.random.seed(seed)
    
    model = None
    if args.train_file:
        vocabx, vocabc, vocaby, maxlw, maxlc = Pipe.create_vocabulary(args.train_file,args.cutoff)
        maxlw=min(maxlw,args.lw_th)
        maxlc=min(maxlc,args.lc_th)
        args.n_out=len(vocaby)
        args.maxlw=maxlw
        args.maxlc=maxlc
        args.vocabx=vocabx
        args.vocabc=vocabc
        args.vocaby=vocaby
        nn = baseNNs(args = args)
        if args.pre_trained:
            embeddings=Pipe.get_W(args.emb_file, args.wsize,vocabx, 0.25)
        else:
            embeddings=None
        model=nn.buildModel(embeddings)
        nn.training(model)
    return model

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    
    argparser.add_argument("--train_file", type = str, default = "../metadata/datasets/train.csv", help = "training file")
    
    argparser.add_argument("--test_file", type = str, default = "../metadata/datasets/test.csv", help = "testing file")
         
    argparser.add_argument("--emb_file", type = str, default = "../metadata/glove.6B/glove.6B.300d.txt", help = "embedding file")
    
    argparser.add_argument("--pre_trained", type = int, default = 1, help = "Use pre-trained embedding or not")

    argparser.add_argument("--trainable",type = int, default = 1, help = "Fine tuning embeddings")
    
    argparser.add_argument("--wsize", type = int, default = 300, help = "word embedding size")
    
    argparser.add_argument("--cwsize", type = int, default = 100, help = "word embedding size learn from character vectors")
    
    argparser.add_argument("--csize", type = int, default = 100, help = "character embedding size")
            
    argparser.add_argument("--cutoff", type = int, default = 2, help = "prune words ocurring <= cutoff")
            
    argparser.add_argument("--learning", type = str, default = "adam", help = "learning method (adagrad, sgd, ...)")
            
    argparser.add_argument("--max_epochs", type = int, default = 100, help = "maximum # of epochs")
    
    argparser.add_argument("--batch_size", type = int, default = 100, help = "mini-batch size")    
        
    argparser.add_argument("--rnndims", type = int, default = 300, help = "fully connected layers")

    argparser.add_argument("--non_linear", type = str, default = "relu", help = "non linear function")
        
    argparser.add_argument("--seed", type = int, default = 3435, help = "random seed of the model")
    
    argparser.add_argument("--dropout_in", type = float, default = 0.5, help = "dropout probability")
    
    argparser.add_argument("--dropout_out", type = float, default = 0.5, help = "dropout probability")
    
    argparser.add_argument("--patience", type = int, default = 4, help = "early stopping")
    
    argparser.add_argument("--lc_th", type = int, default = 32, help = "char threshold")
    
    argparser.add_argument("--lw_th", type = int, default = 1024, help = "word threshold")
    
    args = argparser.parse_args()
    
    model=main(args)


