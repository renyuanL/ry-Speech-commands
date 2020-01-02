# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 23:44:17 2019
@author: renyu

functionalKeras005_spchCmdNmelSpec.py
ryLab00.py

ryTrainModel03.py

"""
# In[]
#import ryPrepareDataset00

# In[]
import numpy as np
import time

basePath= '../ryDatasets/gscV2/'
fn= 'gscV2_data.npz'


t0= time.time()

z= np.load(basePath+fn)

x_train=    z['x_trainWithSil']    
y_train=    z['y_trainWithSil']    
x_val=      z['x_val']      
y_val=      z['y_val']
x_test=     z['x_test']     
y_test=     z['y_test']     

fnModel= 'ryModel.hdf5'

print(".... z= np.load({}) will train into {}".format(fn, fnModel))


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

t0= time.time()

#X_testREAL= get_all_fearure(x_testREAL)
X_test=     get_all_fearure(x_test)
X_val=      get_all_fearure(x_val)
X_train=    get_all_fearure(x_train)

#t0= time.time()
dt= time.time()- t0
print('... get_all_fearure() ... dt(sec)= {:.3f}'.format(dt))

### get_all_fearure() ... dt(sec)= 36.128026723861694
### get_all_fearure() ... dt(sec)= 52.950


# In[]

nTime, nFreq= X_train[0].shape

zzz='''
nTime, nFreq= (125, 128)
'''

# In[]
def normalize(x, axis= None):   
    if axis== None:
        x= (x-x.mean())/x.std()
    else:
        x= (x-x.mean(axis= axis))/x.std(axis= axis)
    
    return x

# In[]
print('.... normalize() ....')

X_train= X_train.reshape(-1, nTime, nFreq, 1).astype('float32') 
X_val=   X_val.reshape(-1, nTime, nFreq, 1).astype('float32') 
X_test=  X_test.reshape( -1, nTime, nFreq, 1).astype('float32') 
#X_testREAL=  X_testREAL.reshape( -1, nTime, nFreq, 1).astype('float32') 

#'''  好像重複做了？！
X_train=     normalize(X_train)#, axis=0)  # normalized for the all set, many utterence
X_val=       normalize(X_val)#, axis=0)
X_test=      normalize(X_test)#, axis=0)
#X_testREAL=  normalize(X_testREAL)#, axis=0)
#'''

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






# In[]
nCategs= len(set(y_train)) #36 #c_train.size #36


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



mc= ModelCheckpoint(fnModel, 
        monitor=    'val_accuracy', 
        verbose=    1, 
        save_best_only= True, 
        mode=      'max')

t0= time.time()

h= m.fit(X_train, y_train,
         
        batch_size=500, #1000, # 1000
        epochs=    100,
        
        callbacks=[es, mc],
        
        #validation_split= 0.1
        validation_data= (X_val, y_val)
        )


#t0= time.time()
dt= time.time()- t0
print('... h= m.fit() ... dt(sec)= {}'.format(dt))

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


# In[]


# In[]

## for version.002

print(' ~~~ simulation session ended ~~~')

# In[]
import numpy as np
from tensorflow.keras.models import load_model
import sounddevice as sd

#import audioUtils


ryGscList=[ 
 '_silence_',
 'one',  'two', 'three', 'four', 'five',
 'six', 'seven', 'eight', 'nine', 'zero',
 'yes', 'no',
 'go', 'stop',
 'on', 'off',
 'up', 'down',
 'left', 'right',
 'forward', 'backward',
 'marvin', 'sheila',
 'dog', 'cat',
 'bird', 'bed',
 'happy', 'house',
 'learn', 'follow',
 'tree', 'visual',
 'wow'
 ]


labels= ryGscList


model= load_model(fnModel)


def predict(x):#, fs=16000):
    prob=  model.predict(x)#.reshape(1,fs,1))
    index= np.argmax(prob[0])
    y= labels[index]
    return y

def recWav(x, featureOut= False):
    x= x.flatten()    

    X= ryFeature(x)['log_mel_spectrogram']
    
    X= X.numpy().astype(np.float32)
    
    X= normalize(X)  # normalized for only one utterence x

    Xin= X.reshape(1,X.shape[0],X.shape[1], 1)
    y= predict(Xin)
    
    if featureOut == True:
        return y, X
    else:
        return y


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
yL= []  
for x, yI in tqdm(zip(x_test[0:1000], y_test[0:1000])): #xL[0:10]:
    #x= x_testREAL[i]
    
    yAns= labels[yI]
    
    x= x.astype(np.float32)
    
    #sd.play(x, samplerate= 16000)
        
    y= recWav(x) #, featureOut=True) 
    
    yL += [y]
    # the acc will be slightly different because of different normalization base
    
    #info= 'n= {:05d}, nWrong= {:05d}, y=【{}】, yAns= [{}]'.format(
    #        n, nWrong, y, yAns)
    #print(info)
    
    
    if y != yAns:
        nWrong += 1
        wrongL += [n]
        
        info= '''n= {:05d}, nWrong= {:05d}, wer= {:.5f}, y=【{}】, yAns= [{}]'''.format(
                 n, nWrong, nWrong/(n+1), y, yAns)
        
        #print(info)
        
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
# In[]    
info= '''n= {:05d}, nWrong= {:05d}, wer= {:.5f}, acc= {:.5f}'''.format(
         n, nWrong, nWrong/n, 1-nWrong/n)

infoL += [info]

print(info)

fnInfo= 'infoL_test.txt'
#np.save(, np.array(infoL))
with open(fnInfo,'w') as fp:
    for info in infoL:
        print(info, file= fp)




# In[]

print('''##########
a Real-time Test..., 
press 【Enter】and speak out within 1 sec      
the words are in the list of 35 words: 

###################
{}
###################

PS: (you cannot say '_silence_', 
it just for "silence" 
or "no sound" 
or "background noise") 
'''.format(labels))


    
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
    
    if y=='stop': break

# In[]
#import pickle
import compress_pickle as cpk

fn='rySp.gz'
cpk.dump(xL, fn)
xL= cpk.load(fn)

# In[]
    
print('... ry: Good Luck ...')










































