# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 21:14:33 2019

@author: renyu
"""

# In[]
import time
import SpeechDownloader
t0= time.time()
gscInfo, nCategs= SpeechDownloader.PrepareGoogleSpeechCmd(version= 2, task= '35word')
dt= time.time()-t0
print('SpeechDownloader.PrepareGoogleSpeechCmd(), dt= {}'.format(dt))
print('gscInfo.keys()= {}, nCategs= {}'.format(gscInfo.keys(), nCategs))

_='''
Downloading http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz 
into sd_GSCmdV2/test.tar.gz
110kKB [00:16, 6.75kKB/s]

Downloading http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz 
into sd_GSCmdV2/train.tar.gz
2.37MKB [05:38, 7.00kKB/s]                                                                                             

Extracting sd_GSCmdV2/test.tar.gz into sd_GSCmdV2/test/
Extracting sd_GSCmdV2/train.tar.gz into sd_GSCmdV2/train/

Converting test set WAVs to numpy files
100%|█████████████████████████████████████████████████████████████████████████████| 4890/4890 [00:29<00:00, 165.65it/s]
Converting training set WAVs to numpy files
100%|█████████████████████████████████████████████████████████████████████████| 105835/105835 [09:02<00:00, 195.21it/s]
Done preparing Google Speech commands dataset version 2
SpeechDownloader.PrepareGoogleSpeechCmd(), 
dt= 1140.5942041873932 sec
'''

# In[]
import SpeechGenerator
'''
class SpeechGen(keras.utils.Sequence):
    """
    'Generates data for Keras'
    
    list_IDs - list of files that this generator should load
    labels - dictionary of corresponding (integer) category to each file in list_IDs
    
    Expects list_IDs and labels to be of the same length
    """
    def __init__(self, list_IDs, labels, batch_size=64, dim=16000, shuffle=True):
'''

batch_size= 64  #100

s= 'train'
trainGen=  SpeechGenerator.SpeechGen(
    gscInfo[s]['files'], 
    gscInfo[s]['labels'],
    batch_size= batch_size #len(gscInfo[s]['labels'])#batch_size
    )

s= 'val'
valGen=  SpeechGenerator.SpeechGen(
    gscInfo[s]['files'], 
    gscInfo[s]['labels'],
    batch_size= batch_size #len(gscInfo[s]['labels']) #batch_size
    )

s= 'test'
testGen=  SpeechGenerator.SpeechGen(
    gscInfo[s]['files'], 
    gscInfo[s]['labels'],
    batch_size= len(gscInfo[s]['labels']),
    shuffle= False
    )

s= 'testREAL'
testREALGen=  SpeechGenerator.SpeechGen(
    gscInfo[s]['files'], 
    gscInfo[s]['labels'],
    batch_size= len(gscInfo[s]['labels']),
    shuffle= False
    )


# In[]

import compress_pickle as cpk
import os

def dump_data(spGenL, fn):
    
    print('dump {} .... '.format(fn))
    
    t0= time.time()
    if not os.path.isfile(fn):
        aL=   []
        for g in spGenL:
            aL+=  [list(g)]
    
        cpk.dump(aL, fn, compression="gzip") 
    
    t1= time.time()
    dt= t1-t0
    print('fn= {}, dt(sec)= {:.3f}'.format(fn, dt))

fn_test=  'google_spcmd_test.gz'
fn_train= 'google_spcmd_train.gz'

if not os.path.isfile(fn_test):
    dump_data([testGen, testREALGen], fn_test)
### fn= google_spcmd_test.gz, dt(sec)= 1905.081

if not os.path.isfile(fn_train):
    dump_data([trainGen, valGen],     fn_train)
### dt(sec)=  2676.6961941719055

# In[]
# In[]

import compress_pickle as cpk
import os
import numpy as np
import time
        
def load_data(fn):
    
    print('load {} .... '.format(fn))
    
    t0= time.time()
    aLL= cpk.load(fn)
    
    def zzz(aL):
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
    
    zLL= []
    for aL in aLL:
        zL= (xL,  yL,  cL)=  zzz(aL)
        zLL += [zL]
    
    t1= time.time()
    dt= t1-t0
    print('fn= {}, dt(sec)= {:.3f}'.format(fn, dt))
    
    return zLL

fn= 'google_spcmd_test.gz'
[(x_test, y_test, c_test),    (x_testREAL, y_testREAL, c_testREAL)]= load_data(fn)

fn= 'google_spcmd_train.gz'
[(x_train, y_train, c_train), (x_val, y_val, c_val)]= load_data(fn)


# In[]
