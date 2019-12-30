# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 23:44:17 2019
@author: renyu

functionalKeras005_spchCmdNmelSpec.py
ryLab00.py

"""
# In[]
#import ryPrepareDataset00

# In[]
import numpy as np

fn= 'ryGsc_sil0.npz'
print(".... z= np.load({}) ....".format(fn))


z= np.load(fn)

x_test=     z['x_test']     
y_test=     z['y_test']     
x_testREAL= z['x_testREAL'] 
y_testREAL= z['y_testREAL'] 
x_train=    z['x_train']    
y_train=    z['y_train']    
x_val=      z['x_val']      
y_val=      z['y_val']

x_testREAL0= x_testREAL[y_testREAL==0]
y_testREAL0= y_testREAL[y_testREAL==0]
    
x_testREAL1= x_testREAL[y_testREAL!=0]
y_testREAL1= y_testREAL[y_testREAL!=0]

# In[]
import tensorflow as tf

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


'''
batch_size= 1000  # 預防 gpu memory 不夠， 分批作業 
x= x_train[0:batch_size].astype(np.float32)
X= ryFeature(x)['log_mel_spectrogram']
X= X.numpy()
'''


# In[]

import time

import tensorflow as tf


def get_all_fearure(all_x, batch_size= 1000):
    t0= time.time()
    
    x= all_x.astype(np.float32)
    
    #batch_size= 1000  # 預防 gpu memory 不夠， 分批作業 
    
    i=0
    XL=[]
    while i < x.shape[0]:
        
        if i+batch_size<=x.shape[0]:
            xx= x[i:i+batch_size]
        else:
            xx= x[i:]
        
        XX= ryFeature(xx)
        X= XX['log_mel_spectrogram'] 
        #'log_spectrogram'] #'mfcc'] #'log_mel_spectrogram']
        
        X= X.numpy().astype(np.float32)
        
        i  += batch_size
        XL += [X]
    
    XL= np.concatenate(XL)
    print('XL.shape={}'.format(XL.shape))
    
    dt= time.time()-t0
    print('tf.signal.stft, 執行時間 dt= {}'.format(dt))
    
    '''
    XL.shape=(64721, 125, 129) # nTime= 16000/128, nFreq=256/2+1
    tf.signal.stft, dt= 8.066392660140991
    '''
    return XL
# In[]
print('.... get_all_fearure() .... ')
X_testREAL= get_all_fearure(x_testREAL)
X_test=     get_all_fearure(x_test)
X_val=      get_all_fearure(x_val)
X_train=    get_all_fearure(x_train)


# In[]

nTime, nFreq= X_train[0].shape

zzz='''
nTime, nFreq= (125, 128)
'''

# In[]
def normalize(x):   
    x= (x-x.mean())/x.std()
    return x

# In[]
print('.... normalize() ....')

X_train= X_train.reshape(-1, nTime, nFreq, 1).astype('float32') 
X_val=   X_val.reshape(-1, nTime, nFreq, 1).astype('float32') 
X_test=  X_test.reshape( -1, nTime, nFreq, 1).astype('float32') 
X_testREAL=  X_testREAL.reshape( -1, nTime, nFreq, 1).astype('float32') 

X_train=     normalize(X_train)
X_val=       normalize(X_val)
X_test=      normalize(X_test)
X_testREAL=  normalize(X_testREAL)


# In[]

import tensorflow as tf

tf.keras.backend.clear_session()  
# For easy reset of notebook state.

from tensorflow              import keras
from tensorflow.keras        import layers, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import AveragePooling1D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# In[]

nCategs= 36 #c_train.size #36


x= Input(shape= (nTime, nFreq, 1))

h= x


#'''
h= Conv2D(8,   (16,16), activation='relu', padding='same')(h)
h= MaxPooling2D((4,4), padding='same')(h)
h= Dropout(0.2)(h)

h= Conv2D(16,   (8,8), activation='relu', padding='same')(h)
h= MaxPooling2D((4,4), padding='same')(h)
h= Dropout(0.2)(h)

h= Flatten()(h)

h= Dense(256,  activation='relu')(h)
h= Dropout(0.2)(h)


h= Dense(nCategs,  activation='softmax')(h)

y= h

m= Model(inputs=  x, 
         outputs= y)

m.summary()



# In[]
#keras.utils.plot_model(m, 'm.png', show_shapes=True)



# In[]
m.compile(  
        loss=    'sparse_categorical_crossentropy',
        metrics= ['accuracy'])


es= EarlyStopping(
        monitor=   'val_loss', 
        min_delta= 1e-10,
        patience=  10, 
        mode=      'min', 
        verbose=   1) 

mc= ModelCheckpoint('ry_best_model.hdf5', 
        monitor=    'val_accuracy', 
        verbose=    1, 
        save_best_only= True, 
        mode=      'max')

h= m.fit(X_train, y_train,
         
        batch_size=1000,
        epochs=    100,
        
        callbacks=[es, mc],
        
        #validation_split= 0.1
        validation_data= (X_val, y_val)
        )

# In[]
import numpy as np
from matplotlib import pyplot as pl
v0= h.history['accuracy']
v1= h.history['val_accuracy']
pl.plot(v0, label='accuracy')
pl.plot(v1, label='val_accuracy')
pl.legend()
pl.grid('on')
pl.show()
#keras.utils.plot_model(m, 'm.png', show_shapes=True)

# In[]


m.evaluate(X_test,      y_test,      verbose=2)

#
# testREAL 裡面，有 408 個  _silence_ 標為 0
# 還有 408 個 各式語音 放在 _unknown_ 也被標示為 0 !!!
#

X_testREAL00= X_testREAL[y_testREAL==0][0:408]
y_testREAL00= y_testREAL[y_testREAL==0][0:408]
    
X_testREAL01= X_testREAL[y_testREAL==0][408:]
y_testREAL01= y_testREAL[y_testREAL==0][408:]

X_testREAL1= X_testREAL[y_testREAL!=0]
y_testREAL1= y_testREAL[y_testREAL!=0]
  

m.evaluate(X_testREAL1, y_testREAL1, verbose=2)   # 這個才算正常的 35 字 中的 大約 10 字
m.evaluate(X_testREAL00, y_testREAL00, verbose=2) # 這個還可看看，_silence_ 的正確率
m.evaluate(X_testREAL01, y_testREAL01, verbose=2) # 這個應該全錯，這叫做 標記錯誤，這小部分要重新安排！！！

zzz='''
Epoch 49/100
83968/84736 [============================>.] - ETA: 0s - loss: 0.3737 - accuracy: 0.8813
Epoch 00049: val_accuracy did not improve from 【0.90919】
84736/84736 [==============================] - 8s 93us/sample - loss: 0.3739 - accuracy: 0.8812 - val_loss: 0.3101 - val_accuracy: 0.9068
Epoch 00049: early stopping

11005/1 - 1s - loss: 0.4222 - accuracy: 【0.8964】
4890/1 - 1s - loss: 46.2492 - accuracy: 【0.7348】 !!!!
 ~~~ simulation session ended ~~~
'''

zzz='''
Epoch 46/100
84000/84736 [============================>.] - ETA: 0s - loss: 0.2507 - accuracy: 0.9196 
Epoch 00046: val_accuracy improved from 0.91508 to 0.91589, saving model to 【ry_best_model.hdf5】
84736/84736 [==============================] - 11s 133us/sample - loss: 0.2507 - accuracy: 0.9196 - val_loss: 0.3050 - val_accuracy: 0.9159
Epoch 47/100
84000/84736 [============================>.] - ETA: 0s - loss: 0.2525 - accuracy: 0.9187 
Epoch 00047: val_accuracy did not improve from 【0.91589】
84736/84736 [==============================] - 11s 133us/sample - loss: 0.2528 - accuracy: 0.9185 - val_loss: 0.3070 - val_accuracy: 0.9128
Epoch 00047: early stopping

11005/1 - 2s - loss: 0.4620 - accuracy: 【0.9070】
4890/1 - 1s - loss: 53.1950 - accuracy: 【0.8084】
'''

'''
11005/1 - 2s - loss: 0.3140 - accuracy: 0.8940

【4074/1】 - 1s - loss: 0.1858 - 【accuracy: 0.8913】
## 去掉 _unknown_ + _silence_= 408+408= 816 個檔案

11005/1 - 1s - loss: 0.5033 - accuracy: 0.9026
4074/1 - 1s - loss: 0.2172 - accuracy: 0.9089

'''

_='''

11005/1 - 1s - loss: 0.5076 - accuracy: 0.8938  <<-- test
4074/1 - 1s - loss: 0.1680 - accuracy: 0.9082   <<-- testREAL1
408/1 - 0s - loss: 0.5170 - accuracy: 0.9338  <<<--- testREAL00，這是 408 個 _silence_，還蠻高的嘛
408/1 - 0s - loss: 55.2395 - accuracy: 0.0000e+00 <<- testREAL01， 標記錯誤，沒辦法。

'''
# In[]


# In[]

## for version.002

print(' ~~~ simulation session ended ~~~')

# In[]
import numpy as np
from tensorflow.keras.models import load_model
import sounddevice as sd

labels= np.array([
        '_silence_', 
        'nine', 
        'yes', 
        'no', 
        'up', 
        'down', 
        'left', 
        'right',
        'on', 
        'off', 
        'stop', 
        'go', 
        'zero', 
        'one', 
        'two', 
        'three', 
        'four',
        'five', 
        'six', 
        'seven', 
        'eight', 
        'backward', 
        'bed', 
        'bird', 
        'cat',
        'dog', 
        'follow', 
        'forward', 
        'happy', 
        'house', 
        'learn', 
        'marvin',
        'sheila', 
        'tree', 
        'visual', 
        'wow'], 
        dtype='<U11')


model= load_model('ry_best_model.hdf5')


def predict(x):#, fs=16000):
    prob=  model.predict(x)#.reshape(1,fs,1))
    index= np.argmax(prob[0])
    y= labels[index]
    return y

def recWav(x, featureOut= False):
    x= x.flatten()    

    X= ryFeature(x)['log_mel_spectrogram']
    
    X= X.numpy().astype(np.float32)
    
    X= normalize(X)

    Xin= X.reshape(1,X.shape[0],X.shape[1], 1)
    y= predict(Xin)
    
    if featureOut == True:
        return y, X
    else:
        return y
# In[]
    
T=  1     # Duration of recording
fs= 16000  # Sample rate

xL= []
for i in range(100):
    
    aKey= input('{}\n{}\n'.format(
                'press "q" to quit', 
                'or another key to record 1 sec speech...'))
    if aKey=='q':
        print('~~~the end~~~')
        break
    
    x= sd.rec(int(T*fs), 
            samplerate= fs, 
            channels= 1, 
            dtype='float32')
        
    sd.wait()  # Wait until recording is finished
    
    y= recWav(x)
    
    print('y= 【{}】'.format(y))

    xL += [x]
# In[]
#import pickle
import compress_pickle as cpk

fn='rySp.gz'
cpk.dump(xL, fn)
xL= cpk.load(fn)


# In[]
    
#import numpy as np
#from tensorflow.keras.models import load_model
import sounddevice as sd

import pylab as pl 
from tqdm import tqdm

n= 0
nWrong= 0 
wrongL= []

infoL= []
  
for x, yI in tqdm(zip(x_testREAL0, y_testREAL0)): #xL[0:10]:
    #x= x_testREAL[i]
    
    yAns= labels[yI]
    
    x= x.astype(np.float32)
    
    #sd.play(x, samplerate= 16000)
        
    y, X= recWav(x, featureOut=True)
    
    #info= 'n= {:05d}, nWrong= {:05d}, y=【{}】, yAns= [{}]'.format(
    #        n, nWrong, y, yAns)
    #print(info)
    
    
    if y != yAns:
        nWrong += 1
        wrongL += [n]
        
        info= '''n= {:05d}, nWrong= {:05d}, wer= {:.5f}, y=【{}】, yAns= [{}]'''.format(
                 n, nWrong, nWrong/(n+1), y, yAns)
        print(info)
        
        infoL += [info]
        
        
        '''
        sd.play(x, samplerate= 16000)
        
        pl.subplot(2,1,1)
        
        pl.title(info)

        pl.imshow(X.transpose(), origin='low')
        
        pl.subplot(2,1,2)
        pl.plot(x)
        pl.grid('on')
        pl.show()
            
        sd.wait()
        '''
    
    n += 1
    #break

np.save('infoL_testREAL.npy', np.array(infoL))

_='''
3966it [02:26, 27.78it/s]n= 03968, nWrong= 00433, wer= 0.10910, y=【two】, yAns= [yes]
4074it [02:30, 27.14it/s]

433/4074
Out[24]: 0.10628375061364752  <--wer for x_testREAL1

'''

_='''
405it [00:17, 17.49it/s]n= 00405, nWrong= 00119, wer= 0.29310, y=【zero】, yAns= [_silence_]
n= 00406, nWrong= 00120, wer= 0.29484, y=【six】, yAns= [_silence_]

【408】it [00:17, 19.53it/s]n= 00408, nWrong= 00121, 【wer= 0.29584】, y=【backward】, yAns= [_silence_]
 <-- wer for x_testREAL0 其中 408 個 真正 的 silence
 
#-----

n= 00815, nWrong= 00528, wer= 0.64706, y=【zero】, yAns= [_silence_]
816it [00:33, 24.65it/s]

528/816
Out[26]: 0.6470588235294118

(528-408)/(816-408)
Out[27]: 0.29411764705882354  <-- wer for x_testREAL0 其中 408 個 真正 的 silence

'''

# In[]

infoL= np.load('infoL_testREAL.npy')
wrongL= [int(infoL[i].split()[1].strip(',')) for i in range(infoL.size)]

for info in infoL[::-1]:
    
    i= int(info.split()[1].strip(','))
    x= x_testREAL[i]
    
    x= x.astype(np.float32)
    
    sd.play(x, samplerate= 16000)
        

    print(info)
            
    pl.title(info)

    pl.plot(x)
    pl.grid('on')
    pl.show()
        
    sd.wait()

    
    
# In[]
    
print('... ry: Good Luck ...')










































