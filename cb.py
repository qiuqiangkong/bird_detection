'''
SUMMARY:  config file
AUTHOR:   Qiuqiang Kong
Created:  2016.11.01
Modified: 2017.01.17    delete unnecessary code
--------------------------------------
'''
import sys
sys.path.append('/user/HS229/qk00006/my_code2015.5-/python/Hat')
from hat.callbacks import Callback
from hat import metrics
import hat.backend as K
from hat.supports import to_list
import config as cfg
import numpy as np
import csv

        
class EvaluationJDC( Callback ):
    # tr_x shoud be 4d, tr_y shoud be 2d
    def __init__( self, tr_x, tr_y, te_x, te_y, batch_size, thres=0.5, call_freq=1 ):
        self._call_freq_ = call_freq
        self._tr_x_ = tr_x
        self._tr_y_ = tr_y
        self._te_x_ = te_x
        self._te_y_ = te_y
        self._batch_size_ = batch_size
        self._thres_ = thres
        
    def compile( self, md ):
        self._md_ = md
        
    def _recognize( self, x, y ):
        [n_songs, n_chunks, n_time, n_freq] = x.shape
        [n_songs, n_out] = y.shape
        
        # frame based err
        [out3d, detect3d] = self._md_.predict( x, self._batch_size_ )     # shape: (n_songs, n_chunks, n_out)
        out2d = out3d.reshape( (n_songs*n_chunks, n_out) )
        y_ex = np.tile( y, n_chunks ).reshape( (n_songs*n_chunks, n_out) )  # shape: (n_songs*n_chunks, n_out)
        err_fb = metrics.binary_error( out2d, y_ex, self._thres_ )
        tp, tn, fp, fn = metrics.tp_tn_fp_fn( out2d, y_ex, self._thres_ )
        print '(frame based):\n', np.array([[tp,fn],[fp,tn]])
        
        # song based err
        y_pred = np.sum( out3d * (detect3d) / ( np.sum(detect3d,axis=1)[:,None,:]+(n_chunks) ), axis=1 )       # shape: (n_songs, n_out)
        err_sb = metrics.binary_error( y_pred, y, self._thres_ )
        tp, tn, fp, fn = metrics.tp_tn_fp_fn( y_pred, y, self._thres_ )
        print '(song based):\n', np.array([[tp,fn],[fp,tn]])
        
        return err_fb, err_sb
        
    def call( self ):
        tr_err_fb, tr_err_sb = self._recognize( self._tr_x_, self._tr_y_ )
        te_err_fb, te_err_sb = self._recognize( self._te_x_, self._te_y_ )
        print 'tr_err (frame based):', tr_err_fb
        print 'te_err (frame based):', te_err_fb
        print 'tr_err (song based):', tr_err_sb
        print 'te_err (song based):', te_err_sb
        