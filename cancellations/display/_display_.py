import os
import math
from re import L
from ..utilities import config as cfg, numutil, tracking,textutil
from ..utilities.tracking import Stopwatch
from ..utilities.setup import session
import collections
from collections import deque
import jax
import jax.numpy as jnp
from ..utilities.textutil import BOX,box,dash,infty
from ..utilities import config as cfg,tracking,textutil, setup
import curses as cs

#----------------------------------------------------------------------------------------------------

class Process(tracking.Process):

	def run_in_display(self,display):
		tracking.loadprocess(self)
		self.display=display
		output=self.execprocess()

		tracking.unloadprocess(self)
		clearscreen()
		return output

	def run_as_main(self):
		def wrapped(screen):
			screen.refresh()
			def getch():
				c=extractkey_cs(screen.getch())
				screen.refresh()
				return c

			setup.screen=screen
			setup.getch=getch
			screen.nodelay(True)
			cs.use_default_colors()
			setup.session.dashboard=_Dashboard_(cs.COLS,cs.LINES)
			return self.run_in_display(setup.session.dashboard)

		return cs.wrapper(wrapped)

	def prepdisplay(self): pass




#----------------------------------------------------------------------------------------------------

def getscreen(): return setup.screen

def clearscreen():
	getscreen().clear()
	getscreen().refresh()

#----------------------------------------------------------------------------------------------------

def BRcrop(text,width=None,height=None):
	lines=text.splitlines()
	if width!=None: lines=[l[:width] for l in lines]
	if height!=None: lines=lines[-height:]
	return '\n'.join(lines)

def TLcrop(text,x0,y0):
	lines=text.splitlines()
	if x0>0: lines=[l[x0:] for l in lines]
	if y0>0: lines=lines[y0:]
	return '\n'.join(lines)

def crop(x,y,text,width,height):
	x_,y_,text_=max(x,0),max(y,0),TLcrop(text,-x,-y)
	width_,height_=width-x_,height-y_
	return x_,y_,BRcrop(text_,width_,height_)

def movingwindow(x,y,text,xlim,ylim):
	x0,x1=xlim
	y0,y1=ylim
	x_,y_,width,height=x-x0,y-y0,x1-x0,y1-y0
	return crop(x_,y_,text,width,height)

#----------------------------------------------------------------------------------------------------



class _Display_:
	output=lambda self: self.getelementstrings()

	def getelementstrings(self):
		return self._getelementstrings_()

	#def _getelementstrings_(self):
	#	return []

	def getfullwidth(self):
		if self._getelementstrings_()==[]: return 0
		return max([x+max([len(l) for l in s.splitlines()]) for x,_,s in self._getelementstrings_()])

	def getfullheight(self):
		if self._getelementstrings_()==[]: return 0
		return max([y+len(s.splitlines()) for _,y,s in self._getelementstrings_()])

	def getwidth(self):
		return self.getfullwidth()

	def getheight(self):
		return self.getfullheight()

delta=lambda x,y: y-x

class _Frame_:
	def __init__(self,width,height,name=None):
		self.height=height
		self.width=width
		self.getcorner=lambda _self_: (0,0)
		self.name=name
		self.outline=False

	def getelementstrings(self):
		strings=self._getelementstrings_()
		x0,y0=self.getcorner(self)
		out=self.movingframe(strings,(x0,x0+self.width),(y0,y0+self.height))

		if self.outline: out+=self.getoutline()
		return out

	@staticmethod
	def movingframe(strings,xlim,ylim):
		return [(x,y,s) for (x,y,S) in strings\
			for (x,y,s) in [movingwindow(x,y,S,xlim,ylim)] if s!='']

	def balign(self):
		self.getcorner=lambda _self_: (0,_self_.getfullheight()-_self_.height)

	def getwidth(self):
		return self.width

	def getheight(self):
		return self.height

	def getoutline(self):
		return [(0,1,self.width*box),(0,self.height,self.width*box),\
			(0,0,'\n'.join(self.height*[box])),(self.width-1,0,'\n'.join(self.height*[box]))]

#----------------------------------------------------------------------------------------------------

class _LinesDisplay_(_Display_):
	def _getelementstrings_(self):
		return [(0,i,l) for i,l in enumerate(self.getlines())]

class _TextDisplay_(_LinesDisplay_):
	def __init__(self,msg,name=None):
		self.msg=msg
		self.name=name

	def gettext(self): return self.msg

	def getlines(self):
		return self.gettext().splitlines()

class _LogDisplay_(_Frame_,_TextDisplay_):
	def __init__(self,process,width,height):
		super().__init__(width,height,name='log')
		self.process=process
		self.balign()

	def gettext(self):
		return '\n'.join(self.process.gethist('recentlog'))

#class _StackedText_(_TextDisplay_):
#	def __init__(self,*elements):
#		self.elements=elements
#
#	def gettext(self):
#		return '\n'.join([e.gettext() for e in self.elements])

#----------------------------------------------------------------------------------------------------


class _CompositeDisplay_(_Display_):
	def __init__(self,*elements,name=None):
		self.elements=list(elements)
		self.name=name

	def add(self,x,y,display):
		self.elements.append((x,y,display))
		return display

	def _getelementstrings_(self):
		out=[]
		for X,Y,e in self.elements:
			out=out+[(X+x,Y+y,s) for x,y,s in e.getelementstrings()]
			if self.name=='T':
				tracking.log(out)
		return out

	#def getelements(self): return self.elements

	def vstack(self,spacing=2):
		E=[]
		y0=0
		for x,y,e in self.elements:
			E.append((x,y0,e))
			y0+=e.getheight()+spacing
		self.elements=E



class _Dashboard_(_Frame_,_CompositeDisplay_):
	def __init__(self,width,height,*elements,name=None,x0=0,y0=0):
		super().__init__(width,height)
		_CompositeDisplay_.__init__(self,*elements)
		self.name=name
		self.x0=x0
		self.y0=y0
		
	def hsplit(self,limits=None,rlimits=[.5],sep=2):
		if limits==None:
			limits=[round(self.width*t) for t in rlimits]
		limits=[0]+limits+[self.width]
		ws=[b-a for a,b in zip(limits[:-1],limits[1:])]

		frames=[_Dashboard_(w-sep,self.height) for w in ws]
		for f,x0 in zip(frames,limits[:-1]):
			self.add(x0,0,f)
		for f,name in zip(frames,['column {}'.format(i) for i in range(1,len(limits))]): f.name=name
		self.fixate(self.x0,self.y0)
		return frames

	def vsplit(self,limits=None,rlimits=.5,sep=1):
		if limits==None:
			limits=[round(self.height*t) for t in rlimits]
		limits=[0]+limits+[self.height]
		hs=[b-a for a,b in zip(limits[:-1],limits[1:])]

		frames=[_Dashboard_(self.width,h-sep) for h in hs]
		for f,y0 in zip(frames,limits[:-1]):
			self.add(0,y0,f)
		for f,name in zip(frames,['row {}'.format(i) for i in range(1,len(limits))]): f.name=name
		self.fixate(self.x0,self.y0)
		return frames

	def fixate(self,x0=0,y0=0):
		self.x0=x0
		self.y0=y0
		for x,y,e in self.elements:
			if isinstance(e,_Dashboard_): e.fixate(x0+x,y0+y)


#----------------------------------------------------------------------------------------------------

	def blankclone(self):
		return _Dashboard_(self.width,self.height,x0=self.x0,y0=self.y0)

#----------------------------------------------------------------------------------------------------
	def arm(self):
		x,y=self.x0,self.y0
		tracking.currentdashboard()[self.name]=cs.newwin(self.getheight()+1,self.getwidth()+1,y,x)

	def draw(self):
		window=tracking.currentdashboard()[self.name]
		window.refresh()

		window.erase()
		for x,y,s in self.getelementstrings():
			window.addstr(y,x,s)
		window.refresh()


#----------------------------------------------------------------------------------------------------
def clearcurrentdash():
	currentdash=tracking.currentdashboard()
	for k,window in currentdash.items():
		window.erase()
		window.clear()
		window.refresh()

	keys=list(currentdash.keys())
	for k in keys:
		del currentdash[k]



#####################################################################################################

def extractkey_cs(a):
    if a>=97 and a<=122: return chr(a)
    if a>=48 and a<=57: return str(a-48)
    match a:
        case 32: return 'SPACE'
        case 10: return 'ENTER'
        case 127: return 'BACKSPACE'
    return a






#----------------------------------------------------------------------------------------------------



halfblocks=[' ',\
	'\u258F',\
	'\u258E',\
	'\u258D',\
	'\u258C',\
	'\u258B',\
	'\u258A',\
	'\u2589',\
	]


def hiresbar(t,width):
	T=t*width
	return BOX*math.floor(T)+halfblocks[math.floor(8*T)%8]


theticks=['\u258F','|','|','\u2595']

def hirestick(t,width,y=0):
	T=t*width
	return (math.floor(T),y,theticks[math.floor(4*T)%4])


theTICKS=['\u2590\u258C ',' \u2588 ',' \u2588 ',' \u2590\u258C']

def hiresTICK(t,width,y=0):
	T=t*width
	return (math.floor(T)-1,y,theTICKS[math.floor(4*T)%4])