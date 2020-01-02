# -*- coding: utf-8 -*-
"""
ryPrepareDataset00.py
ryPrepareDataset02.py
+ryAddSilenceInTrain(allFiles)
"internet" ==> sd_GSCmdV2/ ==> fn= 'ryGsc_sil1.npz'

#--------------------
# ryPrepareDataset03.py
#--------------------

"""
# In[]
from tqdm import tqdm
import requests
import math
import os
import tarfile
import numpy as np
import librosa
import pandas as pd
import time

#import audioUtils

def _downloadFile(url, fName):
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0)); 
    block_size = 1024
    wrote = 0 
    print('Downloading {} into {}'.format(url, fName))
    with open(fName, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), 
                         total= math.ceil(total_size//block_size) , 
                         unit=  'KB', 
                         unit_scale= True):
            
            wrote = wrote  + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")
        
def _extractTar(fname, folder):
    print('Extracting {} into {}'.format(fname, folder))
    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall(path=folder)
        tar.close()
    elif (fname.endswith("tar")):
        tar = tarfile.open(fname, "r:")
        tar.extractall(path=folder)
        tar.close()      

def _DownloadGoogleSpeechCmdV2(basePath, forceDownload= False):
    """
    Downloads Google Speech commands dataset version 2
    """
    #global basePath
    
    dataUrl01= 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
    dataUrl02= 'http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz'
    
    if os.path.isdir(basePath) and not forceDownload:
        print('Google Speech commands dataset version 2 already exists. Skipping download.')
    else:
        if not os.path.exists(basePath):
            os.makedirs(basePath)
        trainFiles= dataUrl01
        testFiles=  dataUrl02
        _downloadFile(testFiles, basePath+'test.tar.gz')
        _downloadFile(trainFiles, basePath+'train.tar.gz')
    
    #extract files
    if not os.path.isdir(basePath+"test/"):
        _extractTar(basePath+'test.tar.gz', basePath+'test/')
        
    if not os.path.isdir(basePath+"train/"):
        _extractTar(basePath+'train.tar.gz', basePath+'train/')
        
#--------------------------        
basePath= '../ryDatasets/gscV2/'
_DownloadGoogleSpeechCmdV2(basePath, forceDownload= False)  
 


# In[]
#read split from files and all files in folders
testWAVs= pd.read_csv(basePath+'train/testing_list.txt', sep=" ", header=None)[0].tolist()
valWAVs=  pd.read_csv(basePath+'train/validation_list.txt', sep=" ", header=None)[0].tolist()


testWAVs= [os.path.join(basePath+'train/', f) for f in testWAVs if f.endswith('.wav')]
valWAVs=  [os.path.join(basePath+'train/', f) for f in valWAVs if f.endswith('.wav')]


allWAVs= []
for root, dirs, files in os.walk(basePath+'train/'):
    allWAVs+= [root+'/'+f  for f in files if f.endswith('.wav')]

trainWAVs= list( set(allWAVs)-set(valWAVs)-set(testWAVs) )

info= '{},{},{},{}'.format(
        len(testWAVs), 
        len(valWAVs), 
        len(trainWAVs), 
        len(allWAVs))
print(info)
# In[]

ryGscDict=  {   'unknown' : 0, 'silence' : 0, 
                '_unknown_' : 0, '_silence_' : 0, 
                '_background_noise_' : 0,
                
                'one' : 1, 'two' : 2, 'three' : 3, 'four' : 4, 'five' : 5,
                'six' : 6, 'seven' : 7,  'eight' : 8, 'nine' : 9,  'zero' : 10,                

                'yes' : 11, 'no' : 12, 
                'go' : 13, 'stop' :14, 
                'on' : 15,  'off' :16, 

                'up' : 17, 'down' : 18, 
                'left' : 19, 'right' : 20,
                'forward':21, 'backward':22, 
                
                'marvin':23,'sheila':24, 
                'dog':25,   'cat':26, 
                'bird':27,  'bed':28, 
                'happy':29, 'house':30,
                'learn':31, 'follow':32,  
                 'tree':33, 'visual':34, 
                 'wow':35}

# list(ryGscCategs.keys())
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

def _getFileCategory(file, catDict):
    """
    Receives a file with name sd_GSCmdV2/train/<cat>/<filename> and returns an integer that is catDict[cat]
    """
    categ = os.path.basename(os.path.dirname(file))
    return catDict.get(categ,0)

#get categories
testWAVlabels= [_getFileCategory(f, ryGscDict) for f in testWAVs]
valWAVlabels=  [_getFileCategory(f, ryGscDict) for f in valWAVs]
trainWAVlabels=[_getFileCategory(f, ryGscDict) for f in trainWAVs]

#background noise should be used for validation as well
bgWAVs= [trainWAVs[i] 
         for i in range(len(trainWAVlabels)) 
         if trainWAVlabels[i]==ryGscDict['silence']]
bgWAVlabels= [ryGscDict['_silence_'] for i in range(len(bgWAVs))]


# In[]
#build dictionaries
testWAVlabelsDict=  dict(zip(testWAVs, testWAVlabels))
valWAVlabelsDict=   dict(zip(valWAVs, valWAVlabels))
trainWAVlabelsDict= dict(zip(trainWAVs, trainWAVlabels))

bgWAVlabelsDict=    dict(zip(bgWAVs, bgWAVlabels))


#info dictionary
trainInfo= {'files': trainWAVs, 'labels' : trainWAVlabelsDict}
valInfo=   {'files': valWAVs,   'labels' : valWAVlabelsDict}
testInfo=  {'files': testWAVs,  'labels' : testWAVlabelsDict}
bgInfo=    {'files': bgWAVs,    'labels' : bgWAVlabelsDict}

gscInfo= {'train': trainInfo, 
          'val':   valInfo, 
          'test':  testInfo,
          'bg':    bgInfo}    

info= [(s, len(gscInfo[s]['files'])) for s in gscInfo.keys()] 
print(info)

# In[]
# In[]
import librosa
import pylab as pl

#import sounddevice as sd

# Data Visualization for 'train'
s='train'

fn= list(gscInfo[s]['labels'].keys())[-1]
x, sr= librosa.load(fn, sr=None)

print(f'x.shape= {x.shape}, sr= {sr}')

pl.figure(figsize=(10,10))
for i in range(100):
    
    fn= list(gscInfo[s]['labels'].keys())[i]
    x, sr= librosa.load(fn, sr=None)
    
    #sd.play(x,sr)

    
    label= list(gscInfo[s]['labels'].values())[i]
    
    c= ryGscList[label]
    
    pl.subplot(10,10,i+1)
    pl.title(c)
    pl.plot(x)
    
    #sd.wait()

# In[]
    
# Data Visualization for 'bg'
    
s='bg'

fn= list(gscInfo[s]['labels'].keys())[-1]
x, sr= librosa.load(fn, sr=None)

print(f'x.shape= {x.shape}, sr= {sr}')


nWav= len(gscInfo[s]['labels'].keys())

pl.figure(figsize=(10,10))
for i in range(nWav):
    
    fn= list(gscInfo[s]['labels'].keys())[i]
    x, sr= librosa.load(fn, sr=None)
    
    #sd.play(x,sr)
    
    label= list(gscInfo[s]['labels'].values())[i]
    
    c= ryGscList[label]
    
    pl.subplot(nWav,1,i+1)
    pl.title(c)
    pl.plot(x)
    
    #sd.wait()
# In[]
# In[]
# load all data into memory
# 有些檔案長度不為 1 sec，要 normalize 成 1 sec= 16000
def ryLengthNormalize(x, length=16000):
    #curX could be bigger or smaller than self.dim
    if len(x) == length:
        X= x
        #print('Same dim')
    elif len(x) > length: #bigger
        #we can choose any position in curX-self.dim
        randPos= np.random.randint(len(x)-length)
        X= x[randPos:randPos+length]
        #print('File dim bigger')
    else: #smaller
        randPos= np.random.randint(length-len(x))
        
        X= np.random.random(length)*1e-10
        
        X[randPos:randPos+len(x)]= x
        #print('File dim smaller')
    return X


# In[]
    
xLL= []
yLL= []

for s in ['val', 'test', 'train']:
    aL=  gscInfo[s]['files']
    xL= []
    for fn in tqdm(aL):
        x, sr= librosa.load(fn, sr= None)
        x= ryLengthNormalize(x)
        xL += [x]
    xL= np.vstack(xL)
    xLL += [xL]
    
    yL=  list(gscInfo[s]['labels'].values())
    yL= np.array(yL)
    yLL += [yL]
    
x_val, x_test, x_train= xLL
y_val, y_test, y_train= yLL


# In[]
# 針對 silence, bg, 長度太長，把他們切成 數個 1秒 sound
bgFiles= gscInfo['bg']['files']

def rySplitSilenceIn1SecSoundList(bgFiles):
    
    noiseL= [librosa.load(fn, sr=None)[0] for fn in bgFiles]
       
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

silenceL= rySplitSilenceIn1SecSoundList(bgFiles)
x_bg= silenceL= np.vstack(silenceL)
y_bg= np.zeros(len(silenceL))

x_trainWithSil=  np.vstack((x_train, x_bg))
y_trainWithSil=  np.concatenate((y_train, y_bg))

# In[]
assert x_train.shape[0]        == y_train.shape[0]
assert x_val.shape[0]          == y_val.shape[0]
assert x_test.shape[0]         == y_test.shape[0]
assert x_trainWithSil.shape[0] == y_trainWithSil.shape[0]

x_trainWithSil= x_trainWithSil.astype('float32')
x_test=         x_test.astype('float32')
x_val=          x_val.astype('float32')

y_trainWithSil= y_trainWithSil.astype('int')
y_test=         y_test.astype('int')
y_val=          y_val.astype('int')


# In[]
import time

t0= time.time()
import os

#basePath= '../ryDatasets/gscV2/'

fn= 'gscV2_data.npz'
if not os.path.isfile(basePath+fn):
    np.savez_compressed(
        basePath+fn, 
        x_trainWithSil=    x_trainWithSil, 
        y_trainWithSil=    y_trainWithSil,
        x_val=      x_val,
        y_val=      y_val,
        x_test=     x_test, 
        y_test=     y_test,
        )

dt= time.time()-t0
print(f'np.savez_compressed(), fn= {fn}, dt(sec)= {dt:.2f}')

# np.savez_compressed(), dt(sec)= 778.8924231529236

# In[]
# In[]
# In[]
# In[]
# In[]
if __name__=='__main__':
    
    t0= time.time()
    
    #basePath= '../ryDatasets/gscV2/'
    fn= 'gscV2_data.npz'
    z= np.load(basePath+fn)
    
    x_train=    z['x_trainWithSil']    
    y_train=    z['y_trainWithSil']    
    x_val=      z['x_val']      
    y_val=      z['y_val']
    x_test=     z['x_test']     
    y_test=     z['y_test']     
    
    
    dt= time.time()-t0
    print(f'np.load({basePath+fn}), dt(sec)= {dt:.3f}')



# In[]
