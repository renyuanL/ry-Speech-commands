'''

ryRealTimeAsr02.py

'''

import sounddevice as sd
import numpy as np

nChannel= 1
nSamplePerSec= sampleRate= 16000
nSamplePerFrame= 1000
nFramePerSec= nSamplePerSec//nSamplePerFrame # 16 frame/sec

indexFrame= 0
bufferTime= 10 #sec
nFramePerBuffer= nFramePerSec* bufferTime #160 # frames == 10 sec
BufferSize= nFramePerBuffer


ryBuffer= (1e-10)*np.random.random((BufferSize, nSamplePerFrame, nChannel))

def ryClearBuffer():
    global ryBuffer, indexFrame
    indexFrame=0
    ryBuffer= (1e-10)*np.random.random((BufferSize, nSamplePerFrame, nChannel))

def ryCallback(indata, outdata, frames, time, status):
    global indexFrame, ryBuffer

    if status:
        print(status)
        
    ## for sound playback
    outdata[:] = indata  # *2

    ryBuffer[indexFrame%BufferSize]= indata       
    indexFrame += 1


import time

import pylab as pl

#import ryLab01_1 as ry
import ryRecog03 as ry


def ryOpenStream(aCallback):
    
    aStream= sd.Stream(callback=    aCallback, 
               channels=   nChannel,       # 1 for mono, 2 for stereo
               samplerate= nSamplePerSec,  # sample/sec
               blocksize=  nSamplePerFrame #1000   # frame_size_in_sample, sample/frame
               )
    return aStream
    

def ryGet1secSpeech(withPrintInfo= False):
    global ryBuffer, BufferSize, indexFrame
    
    x= ryBuffer
    t1= (indexFrame%BufferSize)
    x= np.vstack((x[t1:], x[0:t1]))
    x= x.flatten()    
    x= x.astype(np.float32) 
    x= x[-16000:]    
    spEng= x.std()
    
    if withPrintInfo== True:
        print('[{:.1f}]({:.4f}), '.format(t, spEng), end='', flush=True)
    else:
        print('.', end='', flush=True)
    
    return x

def ryGet1secSpeechAndRecogItWithProb():
    
    x= ryGet1secSpeech()
    
    y, prob= ry.recWav(x, featureOut= False, withProb= True)
    
    return y, prob
        

spDuration= 1000 * bufferTime  # bufferTime= 10 seconds
recProbToConfirm= 0.8
ryClearBuffer()

from tqdm import tqdm

with ryOpenStream(ryCallback) as ryStream:
    
    t0= time.time()
    
    t=   0
    dt= .2 # sec  time_interval to do recognition
    loopNumber= 1000
    loopTime= loopNumber * dt
    
    nStop= 0
    nReallyStop= 10
    
    #while t<spDuration:
    #for i in range(loopNumber):
    print('\n.... Listen forever ....\n')
    while True:
        
        t00= time.time()
        
        y, prob= ryGet1secSpeechAndRecogItWithProb()
        
        if prob > recProbToConfirm: #0.8:
            
            info= f'【{y}】    @time=({t:.1f})'
            print(info, end='\n', flush=True)
        
            if y== 'stop':
                nStop+= 1
                
                info= f'【【{y}】】({nStop}, Really？)'
                print(info, end='\n', flush=True)
                
                if nStop >= nReallyStop:
                    
                    info= f'【【{y}】】(OK, I will STOP！！！)'
                    print(info, end='\n', flush=True)
                    break
            
            if y== 'go':
                nStop= 0
                
                info= f'【【{y}】】(OK, Reset nStop= {nStop}, and then GO)！！！'
                print(info, end='\n', flush=True)
        
        dt00= time.time()-t00
        if dt-dt00>0:
            time.sleep(dt-dt00)
        
        t+=dt
        
    dtt= time.time() - t0
    print('dtt(sec)= {:.3f}'.format(dtt))

print('ry: 明けましておめでとう Happy New Year, 2020 !!!')
