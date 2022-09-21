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
		self.prepdisplay()
		output=self.execprocess()

		tracking.unloadprocess(self)
		self.display.disarm()
		del self.display

		clearscreen()
		return output

	def run_as_main(self):
		def wrapped(screen):
			setup.screen=screen
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

	def _getelementstrings_(self):
		return []

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

	def getelementstrings(self):
		strings=self._getelementstrings_()
		if self.name=='T':
			tracking.log(strings)
		x0,y0=self.getcorner(self)
		out=self.movingframe(strings,(x0,x0+self.width),(y0,y0+self.height))
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

	def getelements(self): return self.elements



class _CompositeFrame_(_Frame_,_CompositeDisplay_):
	def __init__(self,width,height,*elements,name=None):
		super().__init__(width,height)
		_CompositeDisplay_.__init__(self,*elements)
		self.name=name
		
	def hsplit(self,rlimits=[0,.5,1],sep=2):
		limits=[round(self.width*t) for t in rlimits]
		ws=[b-a-sep for a,b in zip(limits[:-1],limits[1:])]
		frames=[_CompositeFrame_(w-sep,self.height) for w in ws]
		for f,x0 in zip(frames,limits[:-1]):
			self.add(x0,0,f)
		for f,name in zip(frames,['column {}'.format(i) for i in range(1,len(limits))]): f.name=name
		return frames

	def vsplit(self,r=.5,sep=2):
		h1=math.floor(self.height*r)
		h2=math.floor(self.height*(1-r))
		frames=(_CompositeFrame_(self.width,h1-sep),_CompositeFrame_(self.width,h2-sep))
		for f,y0 in zip(frames,[0,h1]): self.add(0,y0,f)
		for f,name in zip(frames,['T','B']): f.name=name
		return frames

#----------------------------------------------------------------------------------------------------

class _Dashboard_(_CompositeFrame_):

	def arm(self):
		screencoords=[(e.getheight()+1,e.getwidth()+1,y,x) for x,y,e in self.getelements()]
		self.windows=[cs.newwin(*sc) for sc in screencoords]

	def disarm(self):
		for w in self.windows:
			del w

	def drawall(self):
		for i in range(len(self.getelements())):
			self.draw(i)

	def draw(self,i):
		_e_,w=self.getelements()[i],self.windows[i]

		w.erase()
		X,Y,e=_e_
		for x,y,s in e.getelementstrings():
			w.addstr(y,x,s)
		w.refresh()


#####################################################################################################


def armdashboard(dashboard):
	screencoords=lambda xlim,ylim,*_: (ylim[0],xlim[0],ylim[1],xlim[1])
	tracking.currentdashboard().update({k:cs.newwin(*screencoords(*_e_)) for k,_e_ in dashboard.elements.items()})
