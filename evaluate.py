# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:38:50 2016

@author: duytinvo
"""
from utils.data_utils import Pipe 
from utils.core_nns import saveload
import numpy,sys

def interactive_shell(model,args):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.lower().strip().split(" ")

        if words_raw == ["exit"]:
            break
        
        x=Pipe.wd2idx(words_raw,args.vocabx,args.maxlw,0)
        x=numpy.asarray([x],dtype="int32")
        c=Pipe.ch2idx(words_raw,args.vocabc,args.maxlw,args.maxlc,0)
        c=numpy.asarray([c],dtype="int32")
        y=model.predict([c,x])
        prob=y[0][0]
#        print "********** Probability of Subversion is %.2f. Probability of NonSubversion is %.2f. **********"%(100*(1-prob),100*prob)
        if prob <= 0.5:
            print "\t LABEL: Subversion \t PROB: %.2f"%(100*(1-prob))
        else:
            print "\t LABEL: NonSubversion \t PROB: %.2f"%(100*prob)
        
        

def evaluate(folder,test_file):
    model,args=saveload.load(folder) 
    x_text, c_test, y_test = Pipe.sent2seq(args.test_file, args.vocabx, args.vocabc, args.vocaby, args.maxlw, args.maxlc, 0,0)        

    if args.n_out>2:
        y_test = Pipe.to_categorical(y_test,args.n_out)
    else:
        y_test=numpy.array(y_test)
#    test_size=y_test.shape[0]
    try:
        score, acc = model.evaluate([c_test, x_text], y_test, batch_size=args.batch_size)
        print "\n\tValuating on testing dataset:"
        print '\t\tTest score: %f'% (score)
        print '\t\tTest accuracy: %.2f'% (100*acc)
        print '\t\tTest error: %.2f'% (100*(1-acc))
    except:
        print "Out of memory!!!"
        print "Please try smaller fraction"
    interactive_shell(model,args)
if __name__ == "__main__":
    """
    python evaluate.py ./bilstm/datasets/ ../metadata/datasets/test.csv 
    """
    folder=sys.argv[1]
    test_file=sys.argv[2]
    evaluate(folder,test_file)
