from . import display as disp
from ..utilities import config as cfg,tracking,textutil, setup
import curses as cs






class ConcreteDisplay(disp.DisplayWithDimensions):
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

		self.pad.refresh(0,0,y1,x1,y2-1,x2-1)

	def poke(self,src):
		self.draw()

	def remove(self):
		del self.pad
		super().remove()


class ConcreteStaticTextDisplay(ConcreteDisplay,disp.StaticText): pass

class ConcreteStackedDisplay(ConcreteDisplay,disp.StackedDisplay): pass


class Dashboard(disp.CompositeDisplay):
	def draw(self,*a,**kw):
		for e in self.elements.values():
			e.draw(*a,**kw)



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

setup.checkforinput=checkforinput

def extractkey_cs(a):
    if a>=97 and a<=122: return chr(a)
    if a>=48 and a<=57: return str(a-48)
    match a:
        case 32: return 'SPACE'
        case 10: return 'ENTER'
        case 127: return 'BACKSPACE'
    return a







def getscreen(): return setup.screen

def clearscreen():
	getscreen().clear()
	getscreen().refresh()




class Process(tracking.Process):

	def run_in_display(self,display):
		tracking.loadprocess(self)
		self.display=display
		self.prepdisplay()
		output=self.execprocess()
		tracking.unloadprocess(self)
		self.display.remove()
		clearscreen()
		return output

	def run_as_main(self):
		def wrapped(screen):
			setup.screen=screen
			screen.nodelay(True)
			cs.use_default_colors()
			setup.session.dashboard=Dashboard((0,cs.COLS),(0,cs.LINES))
			return self.run_in_display(setup.session.dashboard)

		return cs.wrapper(wrapped)

	def prepdisplay(self): pass



#def getscreen(): return setup.screen