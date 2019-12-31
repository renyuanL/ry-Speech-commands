# -*- coding: utf-8 -*-
"""
ryPrepareDataset00.py

#--------------------
ryPrepareDataset02.py
#--------------------

+ryAddSilenceInTrain(allFiles)

"internet" ==> sd_GSCmdV2/ ==> fn= 'ryGsc_sil1.npz'



"""

# In[]
import time
import SpeechDownloader

t0= time.time()

gscInfo, nCategs= SpeechDownloader.PrepareGoogleSpeechCmd(
        version= 2, 
        task= '35word')

dt= time.time()-t0
print('SpeechDownloader.PrepareGoogleSpeechCmd(), dt(sec)= {}'.format(dt))

print('gscInfo.keys()= {}, nCategs= {}'.format(gscInfo.keys(), nCategs))


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
    def __init__(self, list_IDs, labels, batch_size=64, 
                 dim=16000, shuffle=True):
'''

batch_size= 1 #64  #100

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
    batch_size= batch_size, #len(gscInfo[s]['labels']) #batch_size
    shuffle= False
    )

s= 'test'
testGen=  SpeechGenerator.SpeechGen(
    gscInfo[s]['files'], 
    gscInfo[s]['labels'],
    batch_size= batch_size, #len(gscInfo[s]['labels']),
    shuffle= False
    )

s= 'testREAL'
testREALGen=  SpeechGenerator.SpeechGen(
    gscInfo[s]['files'], 
    gscInfo[s]['labels'],
    batch_size= batch_size, #len(gscInfo[s]['labels']),
    shuffle= False
    )


#s= 'testREAL'
_='''
trainSilenceGen=  SpeechGenerator.SpeechGen(
    silence_FileList, 
    silence_LabelList,
    batch_size= batch_size, #len(gscInfo[s]['labels']),
    shuffle= False
    )
'''



# In[]
from tqdm import tqdm
import numpy as np

def spGen2xy(spGen):
    
    aL= list(tqdm(spGen))
    
    xL=[]
    yL=[]
    for x,y  in aL:
        xL.append(x)
        yL.append(y)
    x= np.vstack(xL)
    y= np.hstack(yL)
    return x, y

t0= time.time()

x_testREAL, y_testREAL= spGen2xy(testREALGen)
x_test,     y_test=     spGen2xy(testGen)
x_val,      y_val=      spGen2xy(valGen)
x_train,    y_train=    spGen2xy(trainGen)

dt= time.time()-t0
print('spGen2xy(), dt(sec)= {}'.format(dt))

'''
100%|██████████| 4890/4890 [01:27<00:00, 56.12it/s]
100%|██████████| 11005/11005 [04:23<00:00, 41.80it/s]
100%|██████████| 9981/9981 [03:55<00:00, 42.34it/s]
100%|██████████| 85269/85269 [40:14<00:00, 35.32it/s]  
spGen2xy(), dt(sec)= 【3016.8844470977783】
'''

# In[]
#########################
# ry Add silence in Train
#########################
import os
import numpy as np

allFiles= gscInfo['train']['files']
def ryAddSilenceInTrain(allFiles):
    
    noiseL= [np.load(f) for f in allFiles  if '_background_noise_' in f]
       
    n=0
    silenceL= []
    for x in noiseL:
        t=0
        while (t+1)*16000 < x.size:
            x1sec= x[t*16000:(t+1)*16000]
            silenceL += [x1sec]
            t+=1
        n+=1

    return silenceL
        
silenceL= ryAddSilenceInTrain(allFiles)

x_silence= np.vstack(silenceL)
y_silence= np.zeros(len(silenceL)).astype(np.int)

# In[]

x_train= np.vstack([x_train, x_silence])
y_train= np.concatenate([y_train, y_silence])


# In[]
t0= time.time()
import os

fn= 'ryGsc_sil1.npz'
if not os.path.isfile(fn):
    np.savez_compressed(
        fn, 
        x_test=     x_test, 
        y_test=     y_test,
        x_testREAL= x_testREAL,
        y_testREAL= y_testREAL,
        x_train=    x_train, 
        y_train=    y_train,
        x_val=      x_val,
        y_val=      y_val
        )

dt= time.time()-t0
print('spGen2xy(), dt(sec)= {}'.format(dt))
# spGen2xy(), dt(sec)= 778.8924231529236

# In[]
if __name__=='__main__':
    
    t0= time.time()
    
    fn= 'ryGsc_sil1.npz'
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
    
    dt= time.time()-t0
    print('np.load(), dt(sec)= {}'.format(dt))



# In[]
