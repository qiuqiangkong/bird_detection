'''
SUMMARY:  prepare data
AUTHOR:   Qiuqiang Kong
Created:  2016.10.14
Modified: 2017.01.17 delete unnecessary code
--------------------------------------
'''
import sys
sys.path.append('/user/HS229/qk00006/my_code2015.5-/python/Hat')
import numpy as np
from scipy import signal
import cPickle
import os
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import config as cfg
import csv
import wavio
from hat.preprocessing import mat_2d_to_3d, pad_trunc_seqs, enframe
import cPickle
from scipy.fftpack import dct
from sklearn import preprocessing
from sklearn import metrics
from scipy.signal import medfilt

### readwav
def readwav( path ):
    Struct = wavio.read( path )
    wav = Struct.data.astype(float) / np.power(2, Struct.sampwidth*8-1)
    fs = Struct.rate
    return wav, fs

### calculate features
# extract spectrogram feature
def GetSpectrogram( wav_fd, fe_fd, n_delete ):
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.wav') ]
    names = sorted(names)
    cnt = 1
    for na in names:
        print cnt, na
        path = wav_fd + '/' + na
        wav, fs = readwav( path )
        if ( wav.ndim==2 ): 
            wav = np.mean( wav, axis=-1 )
        assert fs==cfg.fs
        ham_win = np.hamming(cfg.win)
        [f, t, X] = signal.spectral.spectrogram( wav, window=ham_win, nperseg=cfg.win, noverlap=0, detrend=False, return_onesided=True, mode='magnitude' ) 
        X = X.T
        X = X[:, n_delete:]

        # DEBUG. print mel-spectrogram
        # plt.matshow(np.log(X.T), origin='lower', aspect='auto')
        # plt.show()

        out_path = fe_fd + '/' + na[0:-4] + '.f'
        cPickle.dump( X, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        cnt += 1


# enhance spectrogram by median filter and substract spectragrm 
def GetEnhanceSpectrogram( fe_fft_fd, out_fe_fd ):
    names = os.listdir( fe_fft_fd )
    names = sorted(names)
    
    cnt = 0
    for na in names:
        if cnt%1==0:
            print cnt, na
            path = fe_fft_fd + '/' + na
            X = cPickle.load( open( path, 'rb' ) )
    
            
            # median filter
            X2 = medfilt( X, kernel_size=(3,7) )
            
            # substract spectrogram
            wid = 50
            X3 = np.zeros_like( X2 )
            
            for i1 in xrange(wid):
                block = X2[:, i1]
                med = np.percentile( block, 25 )
                X3[:, i1] = X2[:,i1] - med * 5
                
            for i1 in xrange(1,(400/wid)):
                block = X2[:, i1*wid:(i1+1)*wid]
                med = np.percentile( block, 25 )
                X3[:, i1] = X2[:,i1] - med * 5
                
            for i1 in xrange( (400/wid), 513 ):
                block = X2[:, i1]
                med = np.percentile( block, 25 )
                X3[:, i1] = X2[:,i1] - med * 5
                
            X3[ np.where( X3<0 ) ] = X2[ np.where(X3<0) ] / 1000
    
            # set low & high frequency to zero
            X3[:,0:10] = 0
            X3[:,450:] = 0
    
            fig, axs = plt.subplots(3)
            axs[0].matshow( np.log(X.T), origin='lower', aspect='auto' )
            axs[1].matshow( np.log(X2.T), origin='lower', aspect='auto' )
            # axs[2].matshow( np.log(X3.T), origin='lower', aspect='auto' )
            plt.show()
            
            out_path = out_fe_fd + '/' + na
            cPickle.dump( X3, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
            
        cnt += 1
     

# get mel feature from spectrogram
def GetMelFromSpectrogram( fe_fft_fd, fe_mel_fd ):
    names = os.listdir( fe_fft_fd )
    names = sorted(names)
    
    cnt = 0
    for na in names:
        print cnt, na
        path = fe_fft_fd + '/' + na
        X = cPickle.load( open( path, 'rb' ) )
        
        if globals().get('melW') is None:
            global melW
            melW = librosa.filters.mel( cfg.fs, n_fft=cfg.win, n_mels=40, fmin=400, fmax=18000 )
            # melW /= np.max(melW, axis=-1)[:,None]
            # plt.plot(melW.T)
            # plt.show()
            
        X2 = np.dot( X, melW.T )
        
        # fig, axs = plt.subplots(2)
        # axs[0].matshow( np.log(X.T), origin='lower', aspect='auto' )
        # axs[1].matshow( np.log(X2.T), origin='lower', aspect='auto' )
        # plt.show()
        
        out_path = fe_mel_fd + '/' + na
        cPickle.dump( X2, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        cnt += 1
        
        
# reduce dimension of spectrogram
def GetPoolSpectrogram( fe_fft_fd, out_fe_fd ):
    names = os.listdir( fe_fft_fd )
    names = sorted(names)
    
    cnt = 0
    for na in names:
        if cnt%1==0:
            print cnt, na
            path = fe_fft_fd + '/' + na
            X = cPickle.load( open( path, 'rb' ) )
    
            # set low & high frequency to zero
            X2 = X
            X2 = X[:, 10:450]
            [n_chunks, n_freq] = X2.shape
            ratio_f = 4
            tmp = np.mean( X2.reshape((n_chunks*(n_freq/ratio_f), ratio_f)), axis=1 ).reshape((n_chunks, n_freq/ratio_f))
            tmp = tmp[0:(n_chunks//2)*2,:]
            [n_chunks, n_freq] = tmp.shape
            tmp2 = np.mean( tmp.T.reshape(((n_chunks/2)*n_freq),2), axis=1 ).reshape((n_freq,n_chunks/2)).T
            
            # fig, axs = plt.subplots(3)
            # axs[0].matshow( np.log(X.T), origin='lower', aspect='auto' )
            # axs[1].matshow( np.log(tmp2.T), origin='lower', aspect='auto' )
            # plt.show()
            X3 = tmp2
            
            out_path = out_fe_fd + '/' + na
            cPickle.dump( X3, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
            
        cnt += 1
        
###
        
def cut_test_fe_tail( mask ):
    num_non_zero = np.sum(mask, axis=1)
    mask[0, num_non_zero-5:] = 0
    return mask
 
           
### Load mini data to speed up
# shape: (n_songs, n_chunks, n_freq)
def GetMiniData3dSongWise_nchunks( fe_fd, cv_csv_path, fold, n_chunks ):
    with open( cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    
    Xall, yall, na_list = [], [], []
    for i1 in xrange( 1, len(lis) ):
        path = fe_fd + '/' + lis[i1][0] + ".f"
        id = int( lis[i1][1] )
        curr_fold = int( lis[i1][2] )
        
        if curr_fold in fold:
            X = cPickle.load( open( path, 'rb' ) )
            Xall.append( X )
            yall += [ id ]
            na_list.append( lis[i1][0] )
            
    n_pad = int( n_chunks )
    Xall, mask = pad_trunc_seqs( np.array( Xall ), n_pad, 'post' )
    yall = np.array( yall )
    
    return Xall, mask, yall, na_list
    
    
# balance positive and negative data
def BalanceData2( X, mask, y ):
    ary0 = np.where(y==0)[0]
    ary1 = np.where(y==1)[0]
    X0 = X[ary0]
    mask0 = mask[ary0]
    X1 = X[ary1]
    mask1 = mask[ary1]
    y0 = y[ary0]
    y1 = y[ary1]
    N0 = len( ary0 )
    N1 = len( ary1 )
    assert N1 > N0
    ratio = int( N1 / N0 )
    if X0.ndim==3: X0_new = np.tile( X0, (ratio,1,1) )
    elif X0.ndim==4: X0_new = np.tile( X0, (ratio,1,1,1) )
    else: Exception('error!!!')
    mask0_new = np.tile( mask0, (ratio,1) )
    y0_new = np.tile( y0, ratio )
    Xall = np.concatenate( [X0_new, X1], axis=0 )
    maskall = np.concatenate( [mask0_new, mask1], axis=0 )
    yall = np.concatenate( [y0_new, y1], axis=0 )
    
    return Xall, maskall, yall


###
def CutBegin3d( X ):
    if X.ndim==2:
        return X[:,10:]
    if X.ndim==3:
        return X[:, 10:, :]


def wipe_click( X3d, na_list ):
    Xall = []
    
    for i1 in xrange( len(X3d) ):
        x2d = X3d[i1,:,:]
        eng = np.mean(x2d, axis=-1)
        

        for i2 in xrange( 1, len(eng)-1 ):
            if eng[i2]/eng[i2-1]>2 and eng[i2]/eng[i2+1]>2:
                x2d[i2] = ( x2d[i2-1] + x2d[i2+1] ) / 2
        
        eng = np.mean(x2d, axis=-1)
        ratio = np.max(eng) / 10
        x2d /= ratio
        Xall.append( x2d )
    
    return np.array( Xall )
        

###
def get_auc( score_ary, gt_ary ):
    N = len( score_ary )
    
    
    tpr_ary = []
    fpr_ary = []
    acc_ary = []
    for thres in np.arange( 0, 1+1e-6, 0.1 ):
        confM = np.zeros((2,2))
        for i1 in xrange( N ):
            if score_ary[i1] > thres:
                y_pred = 1
            else:
                y_pred = 0

            confM[ gt_ary[i1], y_pred ] += 1
        
        tp, fn, fp, tn = confM[0,0], confM[0,1], confM[1,0], confM[1,1]
        tpr = tp / float(tp+fn)
        fpr = fp / float(fp+tn)
        acc = (tp+tn) / float(tp+fn+fp+tn)
        
        tpr_ary.append( tpr )
        fpr_ary.append( fpr )
        acc_ary.append( acc )

    auc = metrics.auc( fpr_ary, tpr_ary )
    return acc_ary, auc
      
        
# create an empty folder
def CreateFolder( fd ):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
###


if __name__ == "__main__":
    assert len( sys.argv )==3, "\nUsage: \npython prepare_data.py arg1 arg2 \narg1: --warblrb | --ff1010 \narg2: --mel | --spectrogram | --denoise_mel | --denoise_mfcc"
    
    if sys.argv[1] == '--warblrb':
        CreateFolder( cfg.wbl_denoise_fe_fd )
        
        if sys.argv[2] == '--denoise_fft':
            CreateFolder( cfg.wbl_denoise_fe_fft_fd )
            GetSpectrogram( cfg.wbl_denoise_wav_fd, cfg.wbl_denoise_fe_fft_fd, n_delete=0 )
        elif sys.argv[2] == '--denoise_enhance_fft':
            CreateFolder( cfg.wbl_denoise_fe_enhance_fft_fd )
            GetEnhanceSpectrogram( cfg.wbl_denoise_fe_fft_fd, cfg.wbl_denoise_fe_enhance_fft_fd )
        elif sys.argv[2] == '--denoise_enhance_mel':
            CreateFolder( cfg.wbl_denoise_fe_enhance_mel_fd )
            GetMelFromSpectrogram( cfg.wbl_denoise_fe_enhance_fft_fd, cfg.wbl_denoise_fe_enhance_mel_fd )
        elif sys.argv[2] == '--denoise_enhance_pool_fft':
            CreateFolder( cfg.wbl_denoise_fe_enhance_pool_fft_fd )
            GetPoolSpectrogram( cfg.wbl_denoise_fe_enhance_fft_fd, cfg.wbl_denoise_fe_enhance_pool_fft_fd )
        else:
            raise Exception( "arg2 incorrect!" )
            
    elif sys.argv[1] == '--test':
        CreateFolder( cfg.test_denoise_fe_fd )

        if sys.argv[2] == '--denoise_fft':
            CreateFolder( cfg.test_denoise_fe_fft_fd )
            GetSpectrogram( cfg.test_denoise_wav_fd, cfg.test_denoise_fe_fft_fd, n_delete=0 )
        elif sys.argv[2] == '--denoise_enhance_fft':
            CreateFolder( cfg.test_denoise_fe_enhance_fft_fd )
            GetEnhanceSpectrogram( cfg.test_denoise_fe_fft_fd, cfg.test_denoise_fe_enhance_fft_fd )
        elif sys.argv[2] == '--denoise_enhance_mel':
            CreateFolder( cfg.test_denoise_fe_enhance_mel_fd )
            GetMelFromSpectrogram( cfg.test_denoise_fe_enhance_fft_fd, cfg.test_denoise_fe_enhance_mel_fd )
        elif sys.argv[2] == '--denoise_enhance_pool_fft':
            CreateFolder( cfg.test_denoise_fe_enhance_pool_fft_fd )
            GetPoolSpectrogram( cfg.test_denoise_fe_enhance_fft_fd, cfg.test_denoise_fe_enhance_pool_fft_fd )
        else:
            raise Exception( "arg2 incorrect!" )
        
    else:
        raise Exception( "arg1 incorrect!" )