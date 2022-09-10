from . import display as disp
from ..utilities import config as cfg,tracking
import curses as cs



class ConcreteDisplay(disp.StackedDisplay):
	def __init__(self,*a,**kw):
		super().__init__(*a,**kw)
		self.pad=cs.newpad(self.height+1,self.width+1)	


	def draw(self):
		self.pad.erase()
		x1,x2=self.xlim
		y1,y2=self.ylim
		lines=self.getlines()
		for i,line in enumerate(lines):
			self.pad.addstr(i,0,line)

		self.pad.refresh(0,0,y1,x1,y2,x2)

	def poke(self,src):
		self.draw()



class Pad(disp.CompositeDisplay):
	def __init__(self,xlim,ylim,WIDTH=None,HEIGHT=None):
		super().__init__(xlim,ylim)
		self.pad=cs.newpad(self.height if HEIGHT==None else HEIGHT,self.width if WIDTH==None else WIDTH)

	def refresh(self,y,x=0):
		x0,x1=self.xlim
		y0,y1=self.ylim
		self.pad.refresh(y,x,y0,x0,y1-1,x1-1)

	def draw(self):
		self.refresh(0,0)

	def addstr(self,*a,**kw): self.pad.addstr(*a,**kw)

	def erase(self): self.pad.erase(); self.pad.clear()





def checkforinput(*args,**kw):
	a=getscreen().getch()
	cs.flushinp()
	tracking.currentprocess().display.draw()
	return extractkey_cs(a)

tracking.checkforinput=checkforinput

def extractkey_cs(a):
    if a>=97 and a<=122: return chr(a)
    if a>=48 and a<=57: return str(a-48)
    match a:
        case 32: return 'SPACE'
        case 10: return 'ENTER'
        case 127: return 'BACKSPACE'
    return a



def getscreen(): return cfg.screen

def clearscreen():
	getscreen().clear()
	getscreen().refresh()

def session_in_display(processfn,profile,nodelay=True,**kw):

	def wrapped(screen):
		cfg.screen=screen
		screen.nodelay(nodelay)
		cs.use_default_colors()
		tracking.session.display=disp.CompositeDisplay((0,cs.COLS),(0,cs.LINES))
		#try: profile.prepdashboard(profile.dashboard)
		#except: pass
		processfn(profile,display=tracking.session.display,**kw)

	out=cs.wrapper(wrapped)
	return out



def runtask(task,profile,display):
	tracking.loadprocess(task)
	output=task(profile,display)
	tracking.unloadprocess(task)
	clearscreen()
	return output

