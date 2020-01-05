'''

ryRealTimeAsr03_pgABC.py

使用 realtime asr 引擎 ryRealTimeAsr03.py

'''

# In[]
#import time

from ryRealTimeAsr03 import ryAsrStream, ryGet1secSpeechAndRecogItWithProb

import pygame as pg
import pygame.time as pgTime

# In[]

pg.init()

aScreen=  pg.display.set_mode((800,200))
pg.display.set_caption('CguAsr2020: a Realtime ASR for SpeechCommands, by Renyuan Lyu.')
#aScreen.fill((0,0,0))

aChineseFont= 'microsoftjhengheimicrosoftjhengheiui'
aFont= pg.font.SysFont(aChineseFont, 20) #'freesansbold.ttf', 32)
#
# 以上中文字體是這樣撈到的...
# 所選的應是【微軟正黑體】
#
'''
>>> pg.font.get_fonts()
Out[15]: 
['arial',
 'arialblack',
 'bahnschrift',
 'calibri',
 :
 'microsoftjhengheimicrosoftjhengheiui',
 'microsoftjhengheimicrosoftjhengheiuibold',
 'microsoftjhengheimicrosoftjhengheiuilight',
 :
'''

aClock= pgTime.Clock()

#
# 語音辨識在此開動 ....，
# 結束時記得 做以下 2 行
# asrStream.stop()
# asrStream.close()
#
asrStream= ryAsrStream()
asrStream.start()



# In[]

#--------------- begin gameLoop-----------

T= 100  # sec
t= 0    # msec = 10**-3 sec
gameLoopRun= True

asrResultList= []
#for i in range(1000):
while t/1000 < T and gameLoopRun == True:
    
    # 用 aClock 來掌握 這個 loop 有多快。
    dt=  aClock.tick(100)  # 100是我設定的上限，實際的速率與電腦的cpu有關
    fps= aClock.get_fps()  # fps 是目前這台機器的速率，約 27 frames/sec
    
    # (Red, Green, Blue)=(0,0,0,) , this is "black"
    aScreen.fill((0,0,0))

    
    # 處理來自玩家之互動【事件】 (event)： 鍵盤、滑鼠 之動作....
    eventList= pg.event.get()
    for event in eventList:
        if event.type == pg.QUIT:
            gameLoopRun= False   # 用滑鼠按視窗右上角 [X]，可終止本迴圈。
    

    # 語音辨識在此執行 ....，
    y, prob= ryGet1secSpeechAndRecogItWithProb()
    
    if prob>.9:
        asrResultList += [(y, prob, t)] # for PostProcessing
    
    info= f'【{y}】, t= {t/1000:.3f}, dt= {dt/1000:.3f}, fps= {fps:.1f}'
    
    if prob>.9: # 要更確認語音辨識結果，只有信心程度(prob)大於 0.9 才有輸出...
        print(info)       
        infoFont= aFont.render(info, True, (255, 255, 255)) 
        # (Red, Green, Blue)=(255,255,255,) , this is "white"
        
        aScreen.blit(infoFont, (0, 100))

    # 每一回合，顯示幕必須更新，這是動畫的基本。
    pg.display.update()
    t += dt

#--- end gameLoop --------------------------------

# In[]

# 運用 相鄰文字取眾數 的方法，
# 對連續的輸出做【平滑】的後處理，
# 可濾掉一些微擾
# 大意是運用一個滑動的文字框(10字)，
# 在文字框中，出現最多者，作為該文字框的代表文字

import numpy as np
import pylab as pl
import scipy.stats as sts

def postProcess(asrResultList):

    asrR= np.vstack(asrResultList)
    
    pl.figure(figsize=(20,5))
    
    x= asrR[:,2].astype('float')
    y= asrR[:,1].astype('float')
    
    #pl.stem(x,y)
    #pl.grid('on')
    #pl.show()
    文字框長= 10      
    z= asrR[:,0]
    nL= []
    for t in range(z.size):
        if t+文字框長 < z.size:
            m= sts.mode(z[t:t+文字框長])
        else:
            m= sts.mode(z[t:])
        n= m.mode
        #if m.count>=10:
        nL += [n]
    
    z= np.vstack(nL).flatten()
    
    asrR1= np.vstack([z, x, y])
       
    zL= [(0, asrR1[:,0])]
    for t in range(asrR1.shape[1]-1):
        if asrR1[0,t+1]!=asrR1[0,t]:
            zL += [(t+1,asrR1[:,t+1])]
    
    zA= np.vstack([a for n, a in zL])
    return zA
    
# zA= postProcess(asrResultList)    
# 運用 相鄰文字取眾數 的方法，
# 對連續的輸出做【平滑】的後處理，
# 可濾掉一些微擾
zA= postProcess(asrResultList)
info= ', '.join(zA[:,0])
print(info)

# In[]    
# 記得語音辨識的機制要善後
asrStream.stop()
asrStream.close()

# pygame 也要善後
input('press any key to quit...')
pg.quit()


# In[]
print('ry: 明けましておめでとう Happy New Year, 2020 !!!')
