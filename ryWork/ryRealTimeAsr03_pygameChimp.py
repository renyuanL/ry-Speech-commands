'''

ryRealTimeAsr03_pygameChimp.py

'''

# In[]

from ryRealTimeAsr03 import ryAsrStream, ryGet1secSpeechAndRecogItWithProb

asrStream= ryAsrStream()
asrStream.start()

#asrStream.stop()
#asrStream.close()


# In[]

#!/usr/bin/env python
"""
https://www.pygame.org/docs/tut/ChimpLineByLine.html?highlight=sprite

This simple example is used for the line-by-line tutorial
that comes with pygame. It is based on a 'popular' web banner.
Note there are comments here, but for the full explanation,
follow along in the tutorial.
"""


# Import Modules
import os, pygame
from pygame.locals import *
from pygame.compat import geterror

if not pygame.font: print('Warning, fonts disabled')
if not pygame.mixer: print('Warning, sound disabled')

main_dir= os.path.split(os.path.abspath(__file__))[0]
data_dir= os.path.join(main_dir, '../ryData')

# In[]
# functions to create our resources
def load_image(name, colorkey=None):
    fullname = os.path.join(data_dir, name)
    try:
        image = pygame.image.load(fullname)
    except pygame.error:
        print('Cannot load image:', fullname)
        raise SystemExit(str(geterror()))
    image = image.convert()
    if colorkey is not None:
        if colorkey is -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, RLEACCEL)
    return image, image.get_rect()


def load_sound(name):
    class NoneSound:
        def play(self): pass
    if not pygame.mixer or not pygame.mixer.get_init():
        return NoneSound()
    fullname = os.path.join(data_dir, name)
    try:
        sound = pygame.mixer.Sound(fullname)
    except pygame.error:
        print('Cannot load sound: %s' % fullname)
        raise SystemExit(str(geterror()))
    return sound


# classes for our game objects
class Fist(pygame.sprite.Sprite):
    """moves a clenched fist on the screen, following the mouse"""
    def __init__(self):
        pygame.sprite.Sprite.__init__(self) #call Sprite initializer
        self.image, self.rect = load_image('fist.bmp', -1)
        self.punching = 0

    def update(self):
        """move the fist based on the mouse position"""
        pos = pygame.mouse.get_pos()
        self.rect.midtop = pos
        if self.punching:
            self.rect.move_ip(5, 10)

    def punch(self, target):
        """returns true if the fist collides with the target"""
        if not self.punching:
            self.punching = 1
            hitbox = self.rect.inflate(-5, -5)
            return hitbox.colliderect(target.rect)

    def unpunch(self):
        """called to pull the fist back"""
        self.punching = 0


class Chimp(pygame.sprite.Sprite):
    """moves a monkey critter across the screen. it can spin the
       monkey when it is punched."""
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)  # call Sprite intializer
        
        self.image, self.rect= load_image('chimp.bmp', -1)
        
        screen= pygame.display.get_surface()
        
        self.area= screen.get_rect()
        self.rect.topleft= 10, 10
        
        self.move= 10 #9 # Chimp 每次 走路(_walk)的距離
        
        self.dizzy= 0 # Chimp 的 暈眩程度

    def update(self):
        """walk or spin, depending on the monkeys state"""
        if self.dizzy:
            self._spin()
        else:
            self._walk()

    def _walk(self):
        """move the monkey across the screen, and turn at the ends"""
        
        newpos= self.rect.move((self.move, 0)) # x方向移動 .move, y方向不動。
        
        # 偵測碰撞左右牆壁，並處理(反彈)
        if not self.area.contains(newpos):
            if self.rect.left < self.area.left or \
                    self.rect.right > self.area.right:
                self.move = -self.move
                newpos = self.rect.move((self.move, 0))
                self.image = pygame.transform.flip(self.image, 1, 0)
            self.rect = newpos

    def _spin(self):
        """spin the monkey image"""
        center= self.rect.center
        self.dizzy= self.dizzy + 10 #12
        if self.dizzy >= 360:
            self.dizzy = 0
            self.image = self.original
        else:
            rotate= pygame.transform.rotate
            self.image= rotate(self.original, self.dizzy)
        self.rect= self.image.get_rect(center= center)

    def punched(self):
        """this will cause the monkey to start spinning"""
        if not self.dizzy:
            self.dizzy= 1
            self.original= self.image


#def main():
"""this function is called when the program starts.
   it initializes everything it needs, then runs in
   a loop until the function returns."""
# Initialize Everything
pygame.init()
screen= pygame.display.set_mode((500, 50))

pygame.display.set_caption('PummelTheChimp,AndWin$$$(揍人猿，贏獎金)')
pygame.mouse.set_visible(0)

# Create The Backgound
background= pygame.Surface(screen.get_size())
background= background.convert()
background.fill((250, 250, 250))

# Put Text On The Background, Centered
if pygame.font:
    
    #font= pygame.font.SysFont(None, 36)
    font= pygame.font.SysFont('microsoftjhengheimicrosoftjhengheiui',12)
    
    '''
    pygame.font.get_fonts()
    Out[23]: 
    ['arial',
     'arialblack',
     'bahnschrift',
     'calibri',
     'cambriacambriamath',
     ....
     (慢慢找到一個..."microsoftjhengheimicrosoftjhengheiui")
     猜它是【微軟正黑體】
    '''
    
    text= font.render("PummelTheChimp,AndWin$$$(揍人猿，贏獎金)", 1, (10, 10, 10))
    textpos= text.get_rect(centerx= background.get_width()/2)
    background.blit(text, textpos)


# Display The Background
screen.blit(background, (0, 0))

'''
blit (plural blits)

(computing) A logical operation 
in which a block of data is rapidly moved or copied in memory, 
most commonly used to animate two-dimensional graphics.
'''

pygame.display.flip()

'''
flip: 本意為翻轉
在此為...
Update the full display Surface to the screen
'''

# Prepare Game Objects
clock= pygame.time.Clock()
whiff_sound= load_sound('whiff.wav')
punch_sound= load_sound('punch.wav')

chimp= Chimp()
fist=  Fist()

allsprites= pygame.sprite.RenderPlain((fist, chimp))

allsprites.update()

# Draw Everything
screen.blit(background, (0, 0))
allsprites.draw(screen)
pygame.display.flip()

# Main Loop
#'''
going= True
while going:
    
    dt= clock.tick(10)  
    # at most 60 calls/sec of clock.tick(60) is allowed
    # dt is the time after the last call of clock.tick()

    # Handle Input Events
    for event in pygame.event.get():
        if event.type == QUIT:
            going= False
        elif event.type == KEYDOWN and event.key == K_ESCAPE:
            going= False
        elif event.type == MOUSEBUTTONDOWN:
            if fist.punch(chimp):
                punch_sound.play()  # punch
                chimp.punched()
            else:
                whiff_sound.play()  # miss
        elif event.type == MOUSEBUTTONUP:
            fist.unpunch()
                       
    # Handle Speech Recognition
    y, prob= ryGet1secSpeechAndRecogItWithProb()
    if y in ['yes','go','marvin'] and prob>.8:
        punch_sound.play()
        chimp.punched()
        
        
        
    allsprites.update()

    # Draw Everything
    screen.blit(background, (0, 0))
    allsprites.draw(screen)
    pygame.display.flip()
#'''
pygame.quit()

asrStream.stop()
asrStream.close()


# In[]

# In[]

print('ry: 明けましておめでとう Happy New Year, 2020 !!!')
