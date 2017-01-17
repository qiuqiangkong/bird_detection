'''
SUMMARY:  config file
AUTHOR:   Qiuqiang Kong
Created:  2017.01.16
Modified: 2017.01.17    delete unnecessary code
--------------------------------------
'''
import sys
sys.path.append('/user/HS229/qk00006/my_code2015.5-/python/Hat')
import numpy as np
np.random.seed(1515)
import os
from hat.models import Sequential, Model
from hat.layers.core import InputLayer, Dense, Dropout, Flatten, Lambda
from hat.callbacks import SaveModel, Validation
from hat.layers.normalization import BN
from hat.preprocessing import sparse_to_categorical, pad_trunc_seqs
from hat import initializations
from hat.layers.cnn import *
from hat.layers.pool import *
from hat.optimizers import SGD, Adam
from hat.regularizations import *
from cb import EvaluationJDC
import matplotlib.pyplot as plt
from hat import serializations
import hat.backend as K
import hat.objectives as obj
import prepare_data as pp_data
import config as cfg
import theano
import theano.tensor as T
import cPickle
import time
from PIL import Image
from sklearn import metrics

n_out = 1
n_hid = 100
tr_fe_fd = cfg.wbl_denoise_fe_enhance_pool_fft_fd
te_fe_fd = cfg.wbl_denoise_fe_enhance_pool_fft_fd

tr_cv_csv_path = cfg.wbl_cv10_csv_path
te_cv_csv_path = cfg.wbl_cv10_csv_path

tr_fold = [0,2,3,4,5,6,7,8,9]
te_fold = [1]

### loss function
def _jdc_loss_func0( md ):
    eps = 1e-6
    a8_node = md.out_nodes_[0]      # shape: (n_songs, n_chunk, n_out)
    b8_node = md.out_nodes_[1]      # shape: (n_songs, n_chunk, n_out)
    gt_node = md.gt_nodes_[0]       # shape: (n_songs, n_out)
    n_chunks = a8_node.shape[1]
    z0_node = md.in_nodes_[1]     # shape: (n_songs, n_chunks)

    a8_node = a8_node * z0_node[:,:,None]
    b8_node = b8_node * z0_node[:,:,None]
    a8_node = T.clip( a8_node, eps, 1-eps )
    b8_node = T.clip( b8_node, eps, 1-eps )     # clip to avoid numerical underflow
    uni_mu = b8_node / K.sum( b8_node, axis=1 )[:,None,:]
    
    jdc_pd = K.sum( a8_node * uni_mu, axis=1 )
    jdc_pd = T.clip( jdc_pd, eps, 1-eps )
    loss_2d = K.binary_crossentropy( jdc_pd, gt_node )

    return K.mean( K.sum( loss_2d, axis=1 ) )


### User defined functions
def _conv2d( input ):
    [n_songs, n_chunks, n_freq] = input.shape
    input4d = input.reshape( (n_songs, 1, n_chunks, n_freq) )
    filter = theano.shared( np.ones((1,1,1,9)).astype(theano.config.floatX) )
    output4d = T.nnet.conv2d( input4d, filter, border_mode=(0,4) )
    output3d = output4d.reshape((n_songs, n_chunks, n_freq))
    return output3d
            
def _reshape_3d_to_4d( input ):
    [n_songs, n_chunks, n_freq] = input.shape
    return input.reshape( (n_songs, 1, n_chunks, n_freq) )
    
def _reshape_4d_to_3d( input ):
    [n_songs, n_fmaps, n_chunks, n_freq] = input.shape
    return input.dimshuffle(0,2,1,3).flatten(3)     # (n_songs, n_chunks, n_maps*n_freq)
    

### Pre load all feature files to a pickle file to speed up. 
def pre_load():
    tr_X, tr_mask, tr_y, tr_na_list = pp_data.GetMiniData3dSongWise_nchunks( tr_fe_fd, tr_cv_csv_path, tr_fold, 220 )    
    te_X, te_mask, te_y, te_na_list = pp_data.GetMiniData3dSongWise_nchunks( te_fe_fd, te_cv_csv_path, te_fold, 220 )
    
    dict = {}
    dict['tr_X'], dict['tr_mask'], dict['tr_y'], dict['tr_na_list'], dict['te_X'], dict['te_mask'], dict['te_y'], dict['te_na_list'] = tr_X, tr_mask, tr_y, tr_na_list, te_X, te_mask, te_y, te_na_list
    
    cPickle.dump( dict, open( cfg.scrap_fd+'/denoise_enhance_pool_fft_all0.p', 'wb' ), protocol=cPickle.HIGHEST_PROTOCOL )
    print "Pre load finished!"


### Train model
def train():
    _loss_func = _jdc_loss_func0
    
    # load data
    t1 = time.time()
    dict = cPickle.load( open( cfg.scrap_fd+'/denoise_enhance_pool_fft_all0.p', 'rb' ) )
    tr_X, tr_mask, tr_y, tr_na_list, te_X, te_mask, te_y, te_na_list = dict['tr_X'], dict['tr_mask'], dict['tr_y'], dict['tr_na_list'], dict['te_X'], dict['te_mask'], dict['te_y'], dict['te_na_list']
    t2 = time.time()
    
    tr_X = pp_data.wipe_click( tr_X, tr_na_list )
    te_X = pp_data.wipe_click( te_X, te_na_list )
    
    # balance data
    tr_X, tr_mask, tr_y = pp_data.BalanceData2( tr_X, tr_mask, tr_y )
    te_X, te_mask, te_y = pp_data.BalanceData2( te_X, te_mask, te_y )
    
    
    print tr_X.shape, tr_y.shape, te_X.shape, te_y.shape
    [n_songs, n_chunks, n_freq] = te_X.shape
    
    tr_y = tr_y.reshape( (len(tr_y), 1) )
    te_y = te_y.reshape( (len(te_y), 1) )
    
    
    # jdc model
    # classifier
    lay_z0 = InputLayer( (n_chunks,) )          # shape:(n_songs, n_chunks) keep the length of songs
    
    lay_in0 = InputLayer( (n_chunks, n_freq), name='in0' )   # shape: (n_songs, n_chunk, n_freq)
    lay_a1 = lay_in0
    # lay_a1 = Lambda( _conv2d )( lay_a1 )
    
    lay_a1 = Lambda( _reshape_3d_to_4d )( lay_a1 )
    lay_a1 = Convolution2D( 32, 3, 3, act='relu', init_type='glorot_uniform', border_mode=(1,1), strides=(1,1), name='a11' )( lay_a1 )
    lay_a1 = Dropout( 0.2 )( lay_a1 )
    lay_a1 = MaxPool2D( pool_size=(1,2) )( lay_a1 )
    
    lay_a1 = Convolution2D( 64, 3, 3, act='relu', init_type='glorot_uniform', border_mode=(1,1), strides=(1,1), name='a12' )( lay_a1 )
    lay_a1 = Dropout( 0.2 )( lay_a1 )
    lay_a1 = MaxPool2D( pool_size=(1,2) )( lay_a1 )
    lay_a1 = Lambda( _reshape_4d_to_3d )( lay_a1 )
    
    lay_a1 = Dense( n_hid, act='relu', name='a2' )( lay_a1 )       # shape: (n_songs, n_chunk, n_hid)
    lay_a1 = Dropout( 0.2 )( lay_a1 )
    lay_a1 = Dense( n_hid, act='relu', name='a4' )( lay_a1 )
    lay_a1 = Dropout( 0.2 )( lay_a1 )
    lay_a1 = Dense( n_hid, act='relu', name='a6' )( lay_a1 )
    lay_a1 = Dropout( 0.2 )( lay_a1 )
    lay_a8 = Dense( n_out, act='sigmoid', init_type='zeros', b_init=0, name='a8' )( lay_a1 )     # shape: (n_songs, n_chunk, n_out)
    
    # detector
    lay_b1 = lay_in0     # shape: (n_songs, n_chunk, n_freq)
    lay_b2 = Lambda( _conv2d )( lay_b1 )    # shape: (n_songs, n_chunk, n_freq)
    lay_b2 = Lambda( _reshape_3d_to_4d )( lay_b1 )
    lay_b2 = MaxPool2D( pool_size=(1,2) )( lay_b2 )
    lay_b2 = Lambda( _reshape_4d_to_3d )( lay_b2 )
    lay_b8 = Dense( n_out, act='hard_sigmoid', init_type='zeros', b_init=-2.3, name='b8' )( lay_b2 )
    md = Model( in_layers=[lay_in0, lay_z0], out_layers=[lay_a8, lay_b8], any_layers=[] )
    
      
    # print summary info of model
    md.summary()

    # callbacks (optional)
    # save model every n epoch (optional)
    pp_data.CreateFolder( cfg.wbl_dev_md_fd )
    pp_data.CreateFolder( cfg.wbl_dev_md_fd+'/cnn_fft' )
    save_model = SaveModel( dump_fd=cfg.wbl_dev_md_fd+'/cnn_fft', call_freq=20, type='iter' )
    validation = Validation( tr_x=None, tr_y=None, va_x=None, va_y=None, te_x=[te_X, te_mask], te_y=te_y, batch_size=100, metrics=[_loss_func], call_freq=20, dump_path=None, type='iter' )
    
    # callbacks function
    callbacks = [save_model, validation]

    
    # EM training
    md.set_gt_nodes( tr_y )
    md.find_layer('a11').set_trainable_params( ['W','b'] )
    md.find_layer('a12').set_trainable_params( ['W','b'] )
    md.find_layer('a2').set_trainable_params( ['W','b'] )
    md.find_layer('a4').set_trainable_params( ['W','b'] )
    md.find_layer('a6').set_trainable_params( ['W','b'] )
    md.find_layer('a8').set_trainable_params( ['W','b'] )
    md.find_layer('b8').set_trainable_params( [] )
    opt_classifier = Adam( 1e-3 )
    f_classify = md.get_optimization_func( loss_func=_loss_func, optimizer=opt_classifier, clip=None )
    
    
    md.find_layer('a11').set_trainable_params( [] )
    md.find_layer('a12').set_trainable_params( [] )
    md.find_layer('a2').set_trainable_params( [] )
    md.find_layer('a4').set_trainable_params( [] )
    md.find_layer('a6').set_trainable_params( [] )
    md.find_layer('a8').set_trainable_params( [] )
    md.find_layer('b8').set_trainable_params( ['W','b'] )
    opt_detector = Adam( 1e-3 )
    f_detector = md.get_optimization_func( loss_func=_loss_func, optimizer=opt_detector, clip=None )
    
    
    _x, _y = md.preprocess_data( [tr_X, tr_mask], tr_y, shuffle=True )
    
    for i1 in xrange(500):
        print '-----------------------'
        opt_classifier.reset()
        md.do_optimization_func_iter_wise( f_classify, _x, _y, batch_size=100, n_iters=80, callbacks=callbacks, verbose=1 )
        print '-----------------------'
        opt_detector.reset()
        md.do_optimization_func_iter_wise( f_detector, _x, _y, batch_size=100, n_iters=20, callbacks=callbacks, verbose=1 )
    

### recognize
def recognize0():
    # load data
    dict = cPickle.load( open( cfg.scrap_fd+'/denoise_enhance_pool_fft_all0.p', 'rb' ) )
    tr_X, tr_mask, tr_y, tr_na_list, te_X, te_mask, te_y, te_na_list = dict['tr_X'], dict['tr_mask'], dict['tr_y'], dict['tr_na_list'], dict['te_X'], dict['te_mask'], dict['te_y'], dict['te_na_list']
    
    tr_X = pp_data.wipe_click( tr_X, tr_na_list )
    te_X = pp_data.wipe_click( te_X, te_na_list )
    
    print tr_X.shape, tr_y.shape, te_X.shape, te_y.shape
    

    x = te_X
    mask = te_mask
    y = te_y
    na_list = te_na_list
    [n_songs, n_chunks,  n_freq] = x.shape
    #K = 10
    K = n_songs
    
    x= x[0:K]
    mask = mask[0:K]

    for epoch in np.arange(1000,5100,1000):
    
        md = serializations.load( cfg.wbl_dev_md_fd+'/cnn_fft/md'+str(epoch)+'_iters.p' )
        
        [out3d, detect3d] = md.predict( [x, mask], batch_size=100 )     # shape: (K, n_chunks, n_out)    
        out3d *= mask[:,:,None]
        detect3d *= mask[:,:,None]
    
        score_ary = []
        gt_ary = []
    
        for i1 in xrange(K):
            uni_mu = detect3d[i1,:,0] / np.sum( detect3d[i1,:,0] )
            score = np.sum( out3d[i1,:,0] * uni_mu )
            
            score_ary.append( score )
            gt_ary.append( y[i1] )
            
            # plot, deubg, DO NOT DELETE!
            # print i1, y[i1], na_list[i1], score, np.sum(out3d[i1,:,0]*detect3d[i1,:,0]), np.sum(detect3d[i1,:,0])
            # 
            # fig, axs = plt.subplots(4, sharex=True)
            # axs[0].matshow( np.log(x[i1,:,:].T), origin='lower', aspect='auto' )
            # axs[0].set_title('mel spectrogram')
            # 
            # axs[1].stem( detect3d[i1,:,0] )
            # axs[1].set_ylim([0,1])
            # axs[1].set_title('detector')
            # 
            # 
            # axs[2].stem( out3d[i1] )
            # axs[2].set_ylim([0,1])
            # axs[2].set_title('classifier')
            # 
            # 
            # axs[3].stem( detect3d[i1,:,0]*out3d[i1,:,0] )
            # axs[3].set_ylim([0,1])
            # axs[3].set_title('overall')
            # plt.show()
    
        acc_ary, auc = pp_data.get_auc( score_ary, gt_ary )
        plt.plot( np.arange( 0, 1+1e-6, 0.1 ), acc_ary, alpha=epoch/float(5000), color='r' )
        plt.axis( [0,1,0,1] )
        print auc
    
    plt.show()
    
    
def recognize_on_test_data():
    # test_fe_fd = cfg.test_denoise_fe_enhance_mel_fd
    test_fe_fd = cfg.test_denoise_fe_enhance_pool_fft_fd
    
    # load data
    md = serializations.load( cfg.wbl_dev_md_fd+'/cnn_fft/md3000_iters.p' )
    
    names = os.listdir( test_fe_fd )
    names = sorted(names)
    i1 = 0

    f = open(cfg.scrap_fd + "/test_bird_result.csv", 'w')
    
    for na in names:
        if i1!=0:
            f.write("\n")
        
        if i1%1==0:
            
            path = test_fe_fd + "/" + na
            X = cPickle.load( open( path, 'rb' ) )
            [n_chunks, n_freq] = X.shape
            #X = pp_data.wipe_click2d( X )
            
            
            X = X.reshape( (1, n_chunks, n_freq) )
            X *=10000   # amplitude test data, which is useful
            
            
            n_pad = int( cfg.n_duration/2 )
            X, mask = pad_trunc_seqs( X, n_pad, 'post' )
            
            mask = pp_data.cut_test_fe_tail( mask )
            X *= mask[:,:,None]
            
            [out3d, detect3d] = md.predict( [X, mask], batch_size=100 )
            out3d *= mask[:,:,None]
            detect3d *= mask[:,:,None]
            
            uni_mu = detect3d[0,:,0] / ( np.sum( detect3d[0,:,0] ) + 1e-8 )
            score = np.sum( out3d[0,:,0] * uni_mu )
            if score < 0.00001: score=0
            
            
            
            print i1, na, score, np.sum(out3d[0,:,0]*detect3d[0,:,0]), np.sum(detect3d[0,:,0])

            # # Plot for debug!
            # fig, axs = plt.subplots(4, sharex=True)
            # axs[0].matshow( np.log(X[0,:,:].T), origin='lower', aspect='auto' )
            # axs[0].set_title('mel spectrogram')
            # 
            # axs[1].stem( detect3d[0,:,0] )
            # axs[1].set_ylim([0,1])
            # axs[1].set_title('detector')
            # 
            # 
            # axs[2].stem( out3d[0] )
            # axs[2].set_ylim([0,1])
            # axs[2].set_title('classifier')
            # 
            # 
            # axs[3].stem( detect3d[0,:,0]*out3d[0,:,0] )
            # axs[3].set_ylim([0,1])
            # axs[3].set_title(score)
            # plt.show()
            

        f.write(na[0:-2] + "," + str(score))
        i1 += 1
    
    f.close()
       
        

### main function
if __name__ == '__main__':
    assert len( sys.argv )==2, "\nUsage: \npython main_dev_dnn.py --train\npython main_dev_dnn.py --recognize"
    if sys.argv[1] == '--pre_load': pre_load()
    if sys.argv[1] == '--train': train()
    if sys.argv[1] == '--recognize0': recognize0()
    if sys.argv[1] == "--recognize_on_test_data": recognize_on_test_data()