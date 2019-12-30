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

#ryBuffer= [None]* BufferSize
#ryBuffer= np.zeros((BufferSize, nSamplePerFrame, nChannel))
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

import ryLab01_1 as ry

spDuration= 10 * bufferTime  # bufferTime= 10 seconds

ryClearBuffer()

with sd.Stream(callback=   ryCallback, 
               channels=   nChannel,       # 1 for mono, 2 for stereo
               samplerate= nSamplePerSec,  # sample/sec
               blocksize=  nSamplePerFrame #1000   # frame_size_in_sample, sample/frame
               ) as ryStream:
    t0= time.time()
    #sd.sleep(int(duration * 1000))
    #time.sleep(spDuration)
    
    t= 0
    while t<spDuration:
    
        print(' {:.1f}, '.format(t), end='', flush=True)
        dt= .2 # sec
        
        '''
        x= ryBuffer.flatten()#.shape
        
        #t2= BufferSize*nSamplePerFrame
        t1= (indexFrame%BufferSize)*nSamplePerFrame
        t0, t1, t2
        y= np.concatenate((x[t1:], x[0:t1]))
        '''
        
        #'''
        x= ryBuffer
        t1= (indexFrame%BufferSize)
        x= np.vstack((x[t1:], x[0:t1]))
        x= x.flatten()
        #'''
        
        #pl.plot(y)
        #pl.show()
        
        x= x.astype(np.float32)
        #xwav, ylabel= ry.rec_long_wav(x= x, dt=.5)
        x= x[-16000:]
        
        spEng= x.std()
        print('({:.4f}), '.format(spEng), end='', flush=True)
        
        y, prob= ry.recWav(x, featureOut= False, withProb= True)
        if prob > 0.8:
            print('【{}】'.format(y), end='\n', flush=True)
                
        time.sleep(dt)
        
        t+=dt
        
    dt= time.time() - t0
    print('dt(sec)= {:.3f}'.format(dt))