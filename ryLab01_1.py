# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 23:44:17 2019
@author: renyu

functionalKeras005_spchCmdNmelSpec.py
ryLab00.py

ryLab01.py   ... 
ryLab01.1.py ...

第一次用 CNN 做出 能辨識 35 個 英文詞 的 語音辨識系統 ...

"""
# In[]


# In[]
#
import time

import numpy as np

import tensorflow as tf

from tensorflow.keras.models import load_model

import sounddevice as sd


# In[]


def ryFeature(x, 
           sample_rate= 16000, 
           
           frame_length= 1024,
           frame_step=    128,  # frame_length//2
           
           num_mel_bins=     128,
           lower_edge_hertz= 20,     # 0
           upper_edge_hertz= 16000/2, # sample_rate/2   
           
           mfcc_dim= 13
           ):
    
    stfts= tf.signal.stft(x, 
                          frame_length, #=  256, #1024, 
                          frame_step, #=    128,
                          #fft_length= 1024
                          pad_end=True
                          )
    
    spectrograms=     tf.abs(stfts)
    log_spectrograms= tf.math.log(spectrograms + 1e-10)
    
    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins= stfts.shape[-1]  #.value
    
    linear_to_mel_weight_matrix= tf.signal.linear_to_mel_weight_matrix(
          num_mel_bins, 
          num_spectrogram_bins, 
          sample_rate, 
          lower_edge_hertz,
          upper_edge_hertz)
    
    mel_spectrograms= tf.tensordot(
          spectrograms, 
          linear_to_mel_weight_matrix, 1)
    
    mel_spectrograms.set_shape(
          spectrograms.shape[:-1].concatenate(
              linear_to_mel_weight_matrix.shape[-1:]))
    
    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms= tf.math.log(mel_spectrograms + 1e-10)
    
    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfccs= tf.signal.mfccs_from_log_mel_spectrograms(
          log_mel_spectrograms)[..., :mfcc_dim]
    
    feature= {'mfcc':               mfccs, 
              'log_mel_spectrogram':log_mel_spectrograms, 
              'log_spectrogram':    log_spectrograms, 
              'spectrogram':        spectrograms}
    
    return  feature


# In[]
def normalize(x):   
    x= (x-x.mean())/x.std()
    return x



# In[]
def predict(x, withProb= False):#, fs=16000):
    global NNmodel, LabelDic

    prob=  NNmodel.predict(x)#.reshape(1,fs,1))
    index= np.argmax(prob[0])
    y= LabelDic[index]
    
    if withProb== True:
        probability= np.max(prob[0])
        
        #y='{} ({:.2f})'.format(y, probability)
        y= (y, probability)
    return y

def recWav(x, featureOut= False, withProb= False):
    x= x.flatten()    

    X= ryFeature(x)['log_mel_spectrogram']
    
    X= X.numpy().astype(np.float32)
    
    X= normalize(X)

    Xin= X.reshape(1,X.shape[0],X.shape[1], 1)
    y=   predict(Xin, withProb)
    
    if featureOut == True:
        return y, X
    else:
        return y





# In[]

def rec_long_wav(x= None, T=1, dt=.5, fs=16000, pauseByKey= False, fn= None):
    
    if pauseByKey==True:
        aKey= input('press a key to record speech...')
    
    if fn==None and x == None:
        x= sd.rec(int(T*fs), 
                samplerate= fs, 
                channels=   1, 
                dtype=      'float32')
            
        sd.wait()  # Wait until recording is finished
        
    elif fn != None:
        x= np.load(fn)
        #T= x.size/fs
    else:
        #T= x.size/fs
        print('x.shape= {}'.format(x.shape))
        pass
    
    T= x.size/fs
    if T==1:
        y= recWav(x)
    elif T>1:
        # 若輸入語音的長度 T > 1 (sec)，
        # 則移動音框 dt 切成一個一個 1 sec 的語音片段
        # 保持 T/dt 個輸出結果 (邊界之處仍有bug...)
        t=0
        yL= []
        while t<T-dt:
            
            if int((1+t)*fs)<=T*fs:
                x1sec= x[int(t*fs) : int(t*fs)+fs]
            else:
                x1sec= np.random.random(1*fs)*1e-10
                x1sec= x1sec.astype(np.float32)
                
                xx= x[int(t*fs): ].flatten()
                x1sec[0:xx.size]= xx 
            
            y= recWav(x1sec, withProb= True)
            yL += [y]
            t += dt
        y= np.array(yL)
    else:
        y= None
        pass
    
    print('y=【{}】'.format(y))
    
    return x, y



LabelDic= {0: '_silence_', 
           1: 'nine', 2: 'yes', 3: 'no', 4: 'up', 5: 'down', 
           6: 'left', 7: 'right', 8: 'on', 9: 'off', 10: 'stop', 
           11: 'go', 12: 'zero', 13: 'one', 14: 'two', 15: 'three', 
           16: 'four', 17: 'five', 18: 'six', 19: 'seven', 20: 'eight', 
           21: 'backward', 22: 'bed', 23: 'bird', 24: 'cat', 25: 'dog', 
           26: 'follow', 27: 'forward', 28: 'happy', 29: 'house', 30: 'learn', 
           31: 'marvin', 32: 'sheila', 33: 'tree', 34: 'visual', 35: 'wow'}

#nCategs= len(LabelDic) #36 

tf.keras.backend.clear_session() 

fnModel= 'ry_best_model1.hdf5'
 
NNmodel= load_model(fnModel)

if __name__=='__main__':
    timeDuration= 10 #sec
    input('press any to record a {} sec wav...'.format(
                timeDuration))
    
    # .... main recognition ....
    x, y= rec_long_wav(T=timeDuration, dt=.1)
    
    xyL= []
    while True:
        aKey= input('press "q" to quit, or any other to record a {} sec wav...'.format(
                timeDuration))
        if aKey == 'q': break
        
        x, y= rec_long_wav(T=timeDuration, dt=.1)
        
        xyL += (x, y)
    
    print('ry: Good Luck, Bye...')    


