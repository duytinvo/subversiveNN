# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 01:26:10 2016

@author: duytinvo
"""
from keras.models import Model, model_from_json
from keras.layers import Dense, Dropout, Embedding, Input, \
                        concatenate, CuDNNLSTM, CuDNNGRU, Reshape, TimeDistributed
import os, gzip
import cPickle as pickle

class saveload:
    @staticmethod            
    def save(model,args, mfile='my_model_architecture.json',wfile='my_model_weights.h5',argfile='my_model_args.pklz'):
        """
        mfile='my_model_architecture.json'
        wfile='my_model_weights.h5'
        argfile='my_model_args.pklz'
        """
        subfolder = os.path.join(args.folder,os.path.basename(os.path.dirname(args.train_file)))
        if not os.path.exists(subfolder): 
            os.mkdir(subfolder)
        json_string = model.to_json()
        open(os.path.join(subfolder,mfile), 'w').write(json_string)
        model.save_weights(os.path.join(subfolder,wfile),overwrite=True)
        with gzip.open(os.path.join(subfolder,argfile), "wb") as fout:
            pickle.dump(args,fout,protocol = pickle.HIGHEST_PROTOCOL)
    @staticmethod
    def load(folder,mfile='my_model_architecture.json',wfile='my_model_weights.h5',argfile='my_model_args.pklz'):
        with gzip.open(os.path.join(folder,argfile), "rb") as fin:
            args = pickle.load(fin)
        model = model_from_json(open(os.path.join(folder,mfile)).read())
        model.load_weights(os.path.join(folder,wfile))
        if args.n_out>2:            
            model.compile(loss='categorical_crossentropy',optimizer=args.learning,metrics=['accuracy'])
        else:
            model.compile(loss='binary_crossentropy',optimizer=args.learning,metrics=['accuracy'])
        return model,args


#------------------------------------------------------------------------------
#                                   BiLSTM
#------------------------------------------------------------------------------
def build_BiRNN(wembeddings,wmax_features, wdims,trainable,cmax_features, cdims,cwdims,
                maxlen,dropout_in,dropout_out,rnndims,n_out,maxlc,mode='lstm'):
    csequences = Input(shape=(maxlen,), dtype='int32')
    cemb=Embedding(cmax_features,
                   cdims,
                   input_length=maxlen)(csequences)
    cembdp=Dropout(dropout_in)(cemb)
    cemb4d=Reshape((maxlen/maxlc,maxlc,cdims))(cembdp)

    if mode=='lstm':
        # apply forwards LSTM
        cfw = TimeDistributed(CuDNNLSTM(cwdims))(cemb4d)
        # apply backwards LSTM
        cbw = TimeDistributed(CuDNNLSTM(cwdims, go_backwards=True))(cemb4d)
    else:
        # apply forwards GRU
        cfw = TimeDistributed(CuDNNGRU(cwdims))(cemb4d)
        # apply backwards RGU
        cbw = TimeDistributed(CuDNNGRU(cwdims, go_backwards=True))(cemb4d)
#    wcemb=TimeDistributed(Bidirectional((CuDNNLSTM(rnndims))))(cemb4d)
    wcemb = concatenate([cfw, cbw], axis=-1)     

    wsequences = Input(shape=(maxlen/maxlc,), dtype='int32')
    if wembeddings is None:
        wemb=Embedding(wmax_features,wdims,input_length=maxlen/maxlc)(wsequences)
    else:
        wemb=Embedding(wmax_features,
                       wdims,
                       weights=[wembeddings],
                       input_length=maxlen/maxlc,
                       trainable=trainable)(wsequences)
    wembdp=Dropout(dropout_in)(wemb)
    wcembs=concatenate([wcemb,wembdp], axis=-1)
    
    if mode=='lstm':
        # apply forwards LSTM
        wfw = CuDNNLSTM(rnndims)(wcembs)
        # apply backwards LSTM
        wbw = CuDNNLSTM(rnndims, go_backwards=True)(wcembs)
    else:
        # apply forwards GRU
        wfw = CuDNNGRU(rnndims)(wcembs)
        # apply backwards RGU
        wbw = CuDNNGRU(rnndims, go_backwards=True)(wcembs)    
    # concatenate the outputs of the 2 LSTMs
    semb = concatenate([wfw, wbw], axis=-1)
    after_dp = Dropout(dropout_out)(semb)
  
    if n_out>2:  
        output = Dense(n_out, activation='softmax')(after_dp)          
    else:
        output = Dense(1, activation='sigmoid')(after_dp) 
    model = Model(inputs=[csequences,wsequences], outputs=output)
    return model
