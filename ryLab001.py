# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 23:44:17 2019
@author: renyu

functionalKeras005_spchCmdNmelSpec.py
ryLab001.py

"""

# In[]

# In[]

import compress_pickle as cpk
import os
import numpy as np

'''        
def get_data_v2(fn= 'spCmdV002_train.gz'):
    
    print('get {} .... '.format(fn))
    
    aL= cpk.load(fn)
    
    xL= []
    yL= []
    for a in aL:
        x, y= a
        xL += [x]
        yL += [y]
    xL= np.concatenate(xL)
    yL= np.concatenate(yL)
    
    cL=  np.array(sorted(list(set(list(yL)))))
    yDist= [list(yL.flatten()).count(c) for c in cL]
    
    
    print('xL.shape= {}\nyL.shape= {}\ncL= {}\nyDist={}'.format(
            xL.shape, yL.shape, cL, yDist))
    
    return xL, yL, cL


data_path= 'spCmdV002_train.gz'
x_train, y_train, c_train= get_data_v2(fn= data_path)

data_path= 'spCmdV002_val.gz'
x_val, y_val, c_val=       get_data_v2(fn= data_path)

data_path= 'spCmdV002_test.gz'
x_test, y_test, c_test=       get_data_v2(fn= data_path)

data_path= 'spCmdV002_testREAL.gz'
x_testREAL, y_testREAL, c_testREAL=       get_data_v2(fn= data_path)
'''

# In[]

# In[]

from ryLab000_PrepareDataset import load_data

fn= 'google_spcmd_test.gz'
[(x_test, y_test, c_test),    (x_testREAL, y_testREAL, c_testREAL)]= load_data(fn)

fn= 'google_spcmd_train.gz'
[(x_train, y_train, c_train), (x_val, y_val, c_val)]= load_data(fn)




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


batch_size= 1000  # 預防 gpu memory 不夠， 分批作業 
x= x_train[0:batch_size].astype(np.float32)
X= ryFeature(x)['log_mel_spectrogram']
X= X.numpy()

zzz= '''

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

nCategs= c_train.size #36


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

m.evaluate(X_test, y_test, verbose=2)
m.evaluate(X_testREAL, y_testREAL, verbose=2)

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


# In[]

## for version.002
'''
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

'''

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


def predict(audio, fs=16000):
    prob= model.predict(audio)#.reshape(1,fs,1))
    index= np.argmax(prob[0])
    return labels[index]
    
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

    x= x.flatten()
    
    X= ryFeature(x)['log_mel_spectrogram']
    
    X= X.numpy().astype(np.float32)
    
    X= normalize(X)

    X= X.reshape(1,X.shape[0],X.shape[1], 1)
    y= predict(X)
    
    print('y= 【{}】'.format(y))
    xL += [x]
# In[]
import pickle

fn='rySp_v2.gz'
cpk.dump(xL, fn)
xL= cpk.load(fn)


# In[]
    
#import numpy as np
#from tensorflow.keras.models import load_model
import sounddevice as sd

import pylab as pl    
for x in xL:
        
    sd.play(x, samplerate= 16000)
    pl.plot(x)
    pl.show()
    
    X= ryFeature(x)['log_mel_spectrogram']
    
    X= X.numpy().astype(np.float32)
    
    
    X= normalize(X)

    Xspec= X.reshape(X.shape[0],X.shape[1])
    pl.imshow(Xspec.transpose(), origin='low')
    pl.show()


    Xin= X.reshape(1,X.shape[0],X.shape[1], 1)
    y= predict(Xin)
    print('y= 【{}】'.format(y))
        
    sd.wait()
    
# In[]
print('... ry: Good Luck ...')

_='''

Done preparing Google Speech commands dataset version 2
SpeechDownloader.PrepareGoogleSpeechCmd(), 【dt= 1696.8315176963806】

gscInfo.keys()= dict_keys(['train', 'test', 'val', 'testREAL']), nCategs= 36
2019-12-27 03:36:30.308800: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_100.dll

dump google_spcmd_test.gz ....
fn= google_spcmd_test.gz, 【dt(sec)= 1649.519】
dump google_spcmd_train.gz ....
fn= google_spcmd_train.gz, 【dt(sec)= 9807.944】

load google_spcmd_test.gz ....
xL.shape= (11005, 16000)
yL.shape= (11005,)
cL= [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
 25 26 27 28 29 30 31 32 33 34 35]
yDist=[408, 419, 405, 425, 406, 412, 396, 396, 402, 411, 402, 418, 399, 424, 405, 400, 445, 394, 406, 408, 165, 207, 185, 194, 220, 172, 155, 203, 191, 161, 195, 212, 193, 165, 206]
xL.shape= (4890, 16000)
yL.shape= (4890,)
cL= [ 0  2  3  4  5  6  7  8  9 10 11]
yDist=[816, 419, 405, 425, 406, 412, 396, 396, 402, 411, 402]
fn= google_spcmd_test.gz, dt(sec)= 7.771
load google_spcmd_train.gz ....
xL.shape= (84800, 16000)
yL.shape= (84800,)
cL= [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35]
yDist=[6, 3168, 3228, 3129, 2944, 3132, 3036, 3016, 3083, 2969, 3109, 3104, 3248, 3140, 3108, 2964, 2954, 3237, 3088, 3200, 3029, 1346, 1594, 1697, 1657, 1711, 1275, 1254, 1632, 1725, 1286, 1710, 1603, 1407, 1287, 1724]
xL.shape= (9920, 16000)
yL.shape= (9920,)
cL= [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
 25 26 27 28 29 30 31 32 33 34 35]
yDist=[353, 394, 404, 350, 375, 347, 362, 359, 369, 349, 370, 384, 350, 343, 353, 367, 364, 372, 387, 345, 150, 212, 182, 180, 197, 131, 145, 217, 194, 127, 194, 203, 159, 139, 193]
fn= google_spcmd_train.gz, dt(sec)= 50.388
load google_spcmd_test.gz ....
xL.shape= (11005, 16000)
yL.shape= (11005,)
cL= [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
 25 26 27 28 29 30 31 32 33 34 35]
yDist=[408, 419, 405, 425, 406, 412, 396, 396, 402, 411, 402, 418, 399, 424, 405, 400, 445, 394, 406, 408, 165, 207, 185, 194, 220, 172, 155, 203, 191, 161, 195, 212, 193, 165, 206]
xL.shape= (4890, 16000)
yL.shape= (4890,)
cL= [ 0  2  3  4  5  6  7  8  9 10 11]
yDist=[816, 419, 405, 425, 406, 412, 396, 396, 402, 411, 402]
fn= google_spcmd_test.gz, 【dt(sec)= 7.698】

load google_spcmd_train.gz ....
xL.shape= (84800, 16000)
yL.shape= (84800,)
cL= [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35]
yDist=[6, 3168, 3228, 3129, 2944, 3132, 3036, 3016, 3083, 2969, 3109, 3104, 3248, 3140, 3108, 2964, 2954, 3237, 3088, 3200, 3029, 1346, 1594, 1697, 1657, 1711, 1275, 1254, 1632, 1725, 1286, 1710, 1603, 1407, 1287, 1724]
xL.shape= (9920, 16000)
yL.shape= (9920,)
cL= [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
 25 26 27 28 29 30 31 32 33 34 35]
yDist=[353, 394, 404, 350, 375, 347, 362, 359, 369, 349, 370, 384, 350, 343, 353, 367, 364, 372, 387, 345, 150, 212, 182, 180, 197, 131, 145, 217, 194, 127, 194, 203, 159, 139, 193]
fn= google_spcmd_train.gz, dt(sec)= 75.980
2019-12-27 06:49:53.806890: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
2019-12-27 06:49:53.877804: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.683
pciBusID: 0000:01:00.0
2019-12-27 06:49:53.883085: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
2019-12-27 06:49:53.887344: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2019-12-27 06:49:53.909947: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2019-12-27 06:49:53.947343: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.683
pciBusID: 0000:01:00.0
2019-12-27 06:49:53.952293: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
2019-12-27 06:49:53.955934: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2019-12-27 06:49:55.494812: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-12-27 06:49:55.498653: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0
2019-12-27 06:49:55.500325: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N
2019-12-27 06:49:55.505390: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 8784 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
2019-12-27 06:49:56.086975: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cufft64_100.dll
2019-12-27 06:49:56.414985: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cublas64_100.dll
XL.shape=(4890, 125, 128)
tf.signal.stft, 執行時間 dt= 1.3792753219604492
XL.shape=(11005, 125, 128)
tf.signal.stft, 執行時間 dt= 3.0988919734954834
XL.shape=(9920, 125, 128)
tf.signal.stft, 執行時間 dt= 1.4850282669067383
XL.shape=(84800, 125, 128)
tf.signal.stft, 執行時間 dt= 21.542152643203735
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 125, 128, 1)]     0
_________________________________________________________________
conv2d (Conv2D)              (None, 125, 128, 8)       2056
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 32, 32, 8)         0
_________________________________________________________________
dropout (Dropout)            (None, 32, 32, 8)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 16)        8208
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 16)          0
_________________________________________________________________
dropout_1 (Dropout)          (None, 8, 8, 16)          0
_________________________________________________________________
flatten (Flatten)            (None, 1024)              0
_________________________________________________________________
dense (Dense)                (None, 256)               262400
_________________________________________________________________
dropout_2 (Dropout)          (None, 256)               0
_________________________________________________________________
dense_1 (Dense)              (None, 36)                9252
=================================================================
Total params: 281,916
Trainable params: 281,916
Non-trainable params: 0
_________________________________________________________________
Train on 84800 samples, validate on 9920 samples
Epoch 1/100
2019-12-27 06:50:37.617241: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudnn64_7.dll
2019-12-27 06:50:39.046347: W tensorflow/stream_executor/cuda/redzone_allocator.cc:312] Internal: Invoking ptxas not supported on Windows
Relying on driver to perform ptx compilation. This message will be only logged once.
84000/84800 [============================>.] - ETA: 0s - loss: 2.6291 - accuracy: 0.2705
Epoch 00001: val_accuracy improved from -inf to 0.53921, saving model to ry_best_model.hdf5
84800/84800 [==============================] - 16s 193us/sample - loss: 2.6224 - accuracy: 0.2722 - val_loss: 1.6802 - val_accuracy: 0.5392
Epoch 2/100
84000/84800 [============================>.] - ETA: 0s - loss: 1.6271 - accuracy: 0.5300
Epoch 00002: val_accuracy improved from 0.53921 to 0.63790, saving model to ry_best_model.hdf5
84800/84800 [==============================] - 13s 148us/sample - loss: 1.6259 - accuracy: 0.5304 - val_loss: 1.2720 - val_accuracy: 0.6379
Epoch 3/100
84000/84800 [============================>.] - ETA: 0s - loss: 1.2436 - accuracy: 0.6393
Epoch 00003: val_accuracy improved from 0.63790 to 0.77984, saving model to ry_best_model.hdf5
84800/84800 [==============================] - 13s 149us/sample - loss: 1.2417 - accuracy: 0.6397 - val_loss: 0.7749 - val_accuracy: 0.7798
Epoch 4/100
84000/84800 [============================>.] - ETA: 0s - loss: 1.0186 - accuracy: 0.6996
Epoch 00004: val_accuracy improved from 0.77984 to 0.80554, saving model to ry_best_model.hdf5
84800/84800 [==============================] - 13s 148us/sample - loss: 1.0186 - accuracy: 0.6997 - val_loss: 0.6859 - val_accuracy: 0.8055
Epoch 5/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.8838 - accuracy: 0.7368
Epoch 00005: val_accuracy improved from 0.80554 to 0.82853, saving model to ry_best_model.hdf5
84800/84800 [==============================] - 12s 146us/sample - loss: 0.8829 - accuracy: 0.7370 - val_loss: 0.6044 - val_accuracy: 0.8285
Epoch 6/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.7787 - accuracy: 0.7674
Epoch 00006: val_accuracy improved from 0.82853 to 0.85595, saving model to ry_best_model.hdf5
84800/84800 [==============================] - 12s 147us/sample - loss: 0.7784 - accuracy: 0.7675 - val_loss: 0.4955 - val_accuracy: 0.8559
Epoch 7/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.7069 - accuracy: 0.7866
Epoch 00007: val_accuracy did not improve from 0.85595
84800/84800 [==============================] - 12s 147us/sample - loss: 0.7073 - accuracy: 0.7866 - val_loss: 0.4975 - val_accuracy: 0.8541
Epoch 8/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.6541 - accuracy: 0.8028
Epoch 00008: val_accuracy improved from 0.85595 to 0.86603, saving model to ry_best_model.hdf5
84800/84800 [==============================] - 12s 147us/sample - loss: 0.6533 - accuracy: 0.8032 - val_loss: 0.4467 - val_accuracy: 0.8660
Epoch 9/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.6045 - accuracy: 0.8166
Epoch 00009: val_accuracy improved from 0.86603 to 0.87450, saving model to ry_best_model.hdf5
84800/84800 [==============================] - 12s 147us/sample - loss: 0.6041 - accuracy: 0.8169 - val_loss: 0.4260 - val_accuracy: 0.8745
Epoch 10/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.5666 - accuracy: 0.8262
Epoch 00010: val_accuracy improved from 0.87450 to 0.88216, saving model to ry_best_model.hdf5
84800/84800 [==============================] - 12s 147us/sample - loss: 0.5664 - accuracy: 0.8264 - val_loss: 0.4059 - val_accuracy: 0.8822
Epoch 11/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.5349 - accuracy: 0.8365
Epoch 00011: val_accuracy improved from 0.88216 to 0.88720, saving model to ry_best_model.hdf5
84800/84800 [==============================] - 12s 147us/sample - loss: 0.5352 - accuracy: 0.8366 - val_loss: 0.3782 - val_accuracy: 0.8872
Epoch 12/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.5110 - accuracy: 0.8447
Epoch 00012: val_accuracy did not improve from 0.88720
84800/84800 [==============================] - 12s 146us/sample - loss: 0.5107 - accuracy: 0.8447 - val_loss: 0.3914 - val_accuracy: 0.8833
Epoch 13/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.4877 - accuracy: 0.8493
Epoch 00013: val_accuracy did not improve from 0.88720
84800/84800 [==============================] - 12s 147us/sample - loss: 0.4878 - accuracy: 0.8493 - val_loss: 0.3979 - val_accuracy: 0.8828
Epoch 14/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.4671 - accuracy: 0.8558
Epoch 00014: val_accuracy improved from 0.88720 to 0.89798, saving model to ry_best_model.hdf5
84800/84800 [==============================] - 12s 147us/sample - loss: 0.4675 - accuracy: 0.8557 - val_loss: 0.3516 - val_accuracy: 0.8980
Epoch 15/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.4503 - accuracy: 0.8617
Epoch 00015: val_accuracy improved from 0.89798 to 0.89829, saving model to ry_best_model.hdf5
84800/84800 [==============================] - 12s 147us/sample - loss: 0.4493 - accuracy: 0.8620 - val_loss: 0.3471 - val_accuracy: 0.8983
Epoch 16/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.4358 - accuracy: 0.8644
Epoch 00016: val_accuracy did not improve from 0.89829
84800/84800 [==============================] - 12s 147us/sample - loss: 0.4359 - accuracy: 0.8644 - val_loss: 0.3547 - val_accuracy: 0.8957
Epoch 17/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.4205 - accuracy: 0.8686
Epoch 00017: val_accuracy improved from 0.89829 to 0.89849, saving model to ry_best_model.hdf5
84800/84800 [==============================] - 12s 147us/sample - loss: 0.4201 - accuracy: 0.8687 - val_loss: 0.3418 - val_accuracy: 0.8985
Epoch 18/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.4087 - accuracy: 0.8732
Epoch 00018: val_accuracy improved from 0.89849 to 0.90081, saving model to ry_best_model.hdf5
84800/84800 [==============================] - 12s 147us/sample - loss: 0.4086 - accuracy: 0.8733 - val_loss: 0.3303 - val_accuracy: 0.9008
Epoch 19/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.3931 - accuracy: 0.8768
Epoch 00019: val_accuracy improved from 0.90081 to 0.90383, saving model to ry_best_model.hdf5
84800/84800 [==============================] - 13s 147us/sample - loss: 0.3929 - accuracy: 0.8769 - val_loss: 0.3280 - val_accuracy: 0.9038
Epoch 20/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.3844 - accuracy: 0.8792
Epoch 00020: val_accuracy did not improve from 0.90383
84800/84800 [==============================] - 12s 147us/sample - loss: 0.3844 - accuracy: 0.8793 - val_loss: 0.3302 - val_accuracy: 0.9001
Epoch 21/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.3737 - accuracy: 0.8829
Epoch 00021: val_accuracy improved from 0.90383 to 0.90746, saving model to ry_best_model.hdf5
84800/84800 [==============================] - 12s 147us/sample - loss: 0.3745 - accuracy: 0.8827 - val_loss: 0.3187 - val_accuracy: 0.9075
Epoch 22/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.3645 - accuracy: 0.8849
Epoch 00022: val_accuracy did not improve from 0.90746
84800/84800 [==============================] - 12s 147us/sample - loss: 0.3645 - accuracy: 0.8849 - val_loss: 0.3273 - val_accuracy: 0.9056
Epoch 23/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.3583 - accuracy: 0.8865
Epoch 00023: val_accuracy did not improve from 0.90746
84800/84800 [==============================] - 12s 147us/sample - loss: 0.3581 - accuracy: 0.8866 - val_loss: 0.3319 - val_accuracy: 0.9017
Epoch 24/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.3516 - accuracy: 0.8904
Epoch 00024: val_accuracy did not improve from 0.90746
84800/84800 [==============================] - 12s 147us/sample - loss: 0.3515 - accuracy: 0.8905 - val_loss: 0.3257 - val_accuracy: 0.9048
Epoch 25/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.3456 - accuracy: 0.8923
Epoch 00025: val_accuracy did not improve from 0.90746
84800/84800 [==============================] - 12s 146us/sample - loss: 0.3457 - accuracy: 0.8923 - val_loss: 0.3275 - val_accuracy: 0.9020
Epoch 26/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.3415 - accuracy: 0.8922
Epoch 00026: val_accuracy improved from 0.90746 to 0.90766, saving model to ry_best_model.hdf5
84800/84800 [==============================] - 12s 147us/sample - loss: 0.3422 - accuracy: 0.8919 - val_loss: 0.3205 - val_accuracy: 0.9077
Epoch 27/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.3282 - accuracy: 0.8959
Epoch 00027: val_accuracy did not improve from 0.90766
84800/84800 [==============================] - 12s 147us/sample - loss: 0.3283 - accuracy: 0.8959 - val_loss: 0.3313 - val_accuracy: 0.9059
Epoch 28/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.3301 - accuracy: 0.8956
Epoch 00028: val_accuracy improved from 0.90766 to 0.91210, saving model to ry_best_model.hdf5
84800/84800 [==============================] - 12s 147us/sample - loss: 0.3294 - accuracy: 0.8958 - val_loss: 0.3097 - val_accuracy: 0.9121
Epoch 29/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.3222 - accuracy: 0.8980
Epoch 00029: val_accuracy did not improve from 0.91210
84800/84800 [==============================] - 12s 147us/sample - loss: 0.3219 - accuracy: 0.8980 - val_loss: 0.3244 - val_accuracy: 0.9072
Epoch 30/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.3162 - accuracy: 0.8996
Epoch 00030: val_accuracy did not improve from 0.91210
84800/84800 [==============================] - 12s 146us/sample - loss: 0.3162 - accuracy: 0.8996 - val_loss: 0.3294 - val_accuracy: 0.9077
Epoch 31/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.3108 - accuracy: 0.9014
Epoch 00031: val_accuracy did not improve from 0.91210
84800/84800 [==============================] - 12s 147us/sample - loss: 0.3111 - accuracy: 0.9013 - val_loss: 0.3159 - val_accuracy: 0.9108
Epoch 32/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.3066 - accuracy: 0.9026
Epoch 00032: val_accuracy did not improve from 0.91210
84800/84800 [==============================] - 12s 147us/sample - loss: 0.3069 - accuracy: 0.9024 - val_loss: 0.3268 - val_accuracy: 0.9053
Epoch 33/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.3033 - accuracy: 0.9044
Epoch 00033: val_accuracy improved from 0.91210 to 0.91391, saving model to ry_best_model.hdf5
84800/84800 [==============================] - 12s 147us/sample - loss: 0.3032 - accuracy: 0.9044 - val_loss: 0.3079 - val_accuracy: 0.9139
Epoch 34/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.2994 - accuracy: 0.9055
Epoch 00034: val_accuracy did not improve from 0.91391
84800/84800 [==============================] - 12s 147us/sample - loss: 0.2992 - accuracy: 0.9055 - val_loss: 0.3205 - val_accuracy: 0.9078
Epoch 35/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.2966 - accuracy: 0.9056
Epoch 00035: val_accuracy did not improve from 0.91391
84800/84800 [==============================] - 12s 147us/sample - loss: 0.2967 - accuracy: 0.9055 - val_loss: 0.3157 - val_accuracy: 0.9102
Epoch 36/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.2877 - accuracy: 0.9085
Epoch 00036: val_accuracy did not improve from 0.91391
84800/84800 [==============================] - 12s 147us/sample - loss: 0.2881 - accuracy: 0.9084 - val_loss: 0.3087 - val_accuracy: 0.9131
Epoch 37/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.2902 - accuracy: 0.9068
Epoch 00037: val_accuracy did not improve from 0.91391
84800/84800 [==============================] - 12s 147us/sample - loss: 0.2903 - accuracy: 0.9067 - val_loss: 0.3180 - val_accuracy: 0.9091
Epoch 38/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.2809 - accuracy: 0.9107
Epoch 00038: val_accuracy did not improve from 0.91391
84800/84800 [==============================] - 12s 147us/sample - loss: 0.2810 - accuracy: 0.9106 - val_loss: 0.3330 - val_accuracy: 0.9060
Epoch 39/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.2800 - accuracy: 0.9105
Epoch 00039: val_accuracy did not improve from 0.91391
84800/84800 [==============================] - 12s 147us/sample - loss: 0.2796 - accuracy: 0.9106 - val_loss: 0.3108 - val_accuracy: 0.9110
Epoch 40/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.2787 - accuracy: 0.9105
Epoch 00040: val_accuracy did not improve from 0.91391
84800/84800 [==============================] - 12s 147us/sample - loss: 0.2793 - accuracy: 0.9104 - val_loss: 0.3282 - val_accuracy: 0.9072
Epoch 41/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.2780 - accuracy: 0.9111
Epoch 00041: val_accuracy did not improve from 0.91391
84800/84800 [==============================] - 12s 147us/sample - loss: 0.2779 - accuracy: 0.9111 - val_loss: 0.3211 - val_accuracy: 0.9099
Epoch 42/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.2737 - accuracy: 0.9128
Epoch 00042: val_accuracy did not improve from 0.91391
84800/84800 [==============================] - 12s 146us/sample - loss: 0.2736 - accuracy: 0.9129 - val_loss: 0.3100 - val_accuracy: 0.9126
Epoch 43/100
84000/84800 [============================>.] - ETA: 0s - loss: 0.2672 - accuracy: 0.9135
Epoch 00043: val_accuracy did not improve from 0.91391
84800/84800 [==============================] - 12s 147us/sample - loss: 0.2670 - accuracy: 0.9137 - val_loss: 0.3279 - val_accuracy: 0.9088
Epoch 00043: early stopping

11005/1 - 1s - loss: 0.4285 - accuracy: 【【0.9016】】
4890/1 - 1s - loss: 59.3410 - accuracy: 【【0.7957】】

 ~~~ simulation session ended ~~~
press "q" to quit
or another key to record 1 sec speech...

y= 【zero】
press "q" to quit
or another key to record 1 sec speech...

y= 【one】
press "q" to quit
or another key to record 1 sec speech...

y= 【two】
press "q" to quit
or another key to record 1 sec speech...

y= 【three】
press "q" to quit
or another key to record 1 sec speech...

y= 【four】
press "q" to quit
or another key to record 1 sec speech...

y= 【five】
press "q" to quit
or another key to record 1 sec speech...

y= 【six】
press "q" to quit
or another key to record 1 sec speech...

y= 【seven】
press "q" to quit
or another key to record 1 sec speech...

y= 【eight】
press "q" to quit
or another key to record 1 sec speech...

y= 【nine】
press "q" to quit
or another key to record 1 sec speech...

y= 【up】
press "q" to quit
or another key to record 1 sec speech...

y= 【down】
press "q" to quit
or another key to record 1 sec speech...

y= 【left】
press "q" to quit
or another key to record 1 sec speech...

y= 【right】
press "q" to quit
or another key to record 1 sec speech...

y= 【forward】
press "q" to quit
or another key to record 1 sec speech...

y= 【backward】
press "q" to quit
or another key to record 1 sec speech...

y= 【yes】
press "q" to quit
or another key to record 1 sec speech...

y= 【no】
press "q" to quit
or another key to record 1 sec speech...

y= 【stop】
press "q" to quit
or another key to record 1 sec speech...

y= 【go】
press "q" to quit
or another key to record 1 sec speech...

y= 【on】
press "q" to quit
or another key to record 1 sec speech...

y= 【off】
press "q" to quit
or another key to record 1 sec speech...

y= 【bird】
press "q" to quit
or another key to record 1 sec speech...

y= 【bed】
press "q" to quit
or another key to record 1 sec speech...

y= 【house】
press "q" to quit
or another key to record 1 sec speech...

y= 【happy】
press "q" to quit
or another key to record 1 sec speech...

y= 【dog】
press "q" to quit
or another key to record 1 sec speech...

y= 【cat】
press "q" to quit
or another key to record 1 sec speech...

y= 【marvin】
press "q" to quit
or another key to record 1 sec speech...

y= 【sheila】
press "q" to quit
or another key to record 1 sec speech...
q
~~~the end~~~
y= 【zero】
y= 【one】
y= 【two】
y= 【three】
y= 【four】
y= 【five】
y= 【six】
y= 【seven】
y= 【eight】
y= 【nine】
y= 【up】
y= 【down】
y= 【left】
y= 【right】
y= 【forward】
y= 【backward】
y= 【yes】
y= 【no】
y= 【stop】
y= 【go】
y= 【on】
y= 【off】
y= 【bird】
y= 【bed】
y= 【house】
y= 【happy】
y= 【dog】
y= 【cat】
y= 【marvin】
y= 【sheila】
... ry: Good Luck ...

E:\OneDrive\__ryTeach\_2019\SpeechRecognition\__exp1__>


'''









































