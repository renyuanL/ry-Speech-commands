import argparse
import math
import shutil

import numpy as np
import sounddevice as sd



# In[]

import numpy as np
import pylab as pl

import ryLab01_1 as ry

import time

try:
    #samplerate= sd.query_devices(args.device, 'input')['default_samplerate']
    
    '''
    sd.query_devices(args.device, 'input')['default_samplerate']
    Out[8]: 44100.0
    '''
    samplerate= 16000
    

    frameIndex= 0
    nFrames= 100  # T= 100*0.1= 10 sec
    frameSize= int(0.1*samplerate) # 0.1 sec # 160
    spBuffer= np.zeros((nFrames, frameSize))  # 100*0.1//16000= 
    
    
    def callback(indata, frames, time, status):
        global frameIndex, nFrames, spBuffer
        
        if status:
            text = ' ' + str(status) + ' '
            print('{}'.format(status), sep='.')
    
        if any(indata):                ### indata[]，在此偵測
            

            #y=     indata.mean()
            #sigma= indata.std()
            
            #print('frameIndex= [{}], indata.shape= {}, (mean,std)= {}, {}'.format(
            #        frameIndex, indata.shape, y, sigma), 
            #      sep= '')
            
            if frameIndex%10==0:
                print('.{}.'.format(frameIndex) , end='', flush=True)
            
            spBuffer[frameIndex%nFrames]= indata.flatten()
            
            frameIndex += 1
            
        else:
            print('... no input ...')
    
   

    
    with sd.InputStream(#device=     args.device, 
                        channels=   1, 
                        callback=   callback,
                        blocksize=  frameSize, #1024, #int(samplerate * args.block_duration / 1000),
                        samplerate= samplerate):
        while True:
            
            '''
            aKey= input()
            if aKey in ('q', 'Q'):
                break
            '''
            if frameIndex >= 1000:
                break
            
            i= frameIndex%100
            i0= i-10
            if i0>=0:
                x= spBuffer[i0:i]
            elif i0>=0 and i==0:
                x= spBuffer[i0:]
            else:
                x= np.concatenate([spBuffer[i0:], spBuffer[:i]])
            x= x.flatten()
            #pl.plot(x)
            #pl.show()
            
            if frameIndex>10:
                x= x.astype(np.float32)
                #xwav, ylabel= ry.rec_long_wav(x= x, dt=.5)
                y, prob= ry.recWav(x, featureOut= False, withProb= True)
                if prob > 0.8:
                    print('【{}】'.format(y), end='', flush=True)
            
            time.sleep(0.1)            

            


except KeyboardInterrupt:
    parser.exit('Interrupted by user')

except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))

# In[]