import display as disp
import pdb
import config as cfg
import numpy as np
import sys
from config import session


import curses as cs




class ConcreteDisplay(disp.StackedDisplay):
	def __init__(self,xlim,ylim,**kw):
		super().__init__(**kw)
		self.xlim=xlim
		self.ylim=ylim
		x0,x1=self.xlim
		y0,y1=self.ylim
		self.width=x1-x0
		self.height=y1-y0
		self.pad=cs.newpad(self.height,self.width)	


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



def getwrapped(runfn):

	def wrapped(screen):

		cs.use_default_colors()
		cfg.screen=screen
		cfg.dashboard=disp.CDashboard(width=cs.COLS,height=cs.LINES)

		screen.nodelay(True)

		def getinput(*args,**kw):
			a=screen.getch()
			cs.flushinp()
			cfg.dashboard.draw()
			return cfg.extractkey_cs(a)

		cfg.getinput=getinput

		cfg.prepdashboard()
		runfn()
	return wrapped






def RID(runfn,*x,**y):
	wrapped=getwrapped(runfn,*x,**y)
	cs.wrapper(wrapped)




# test
#
#if __name__=='__main__':
#
#	import time
#
#	def wrapped(screen):
#		#cs.use_default_colors()
#		h=cs.LINES
#		w=cs.COLS
#
#		dashboard=CDashboard(50,20)
#
#		S=db.StackedDisplay(memory=session,width=w)
#		C=dashboard.add(S,(0,w-1),(0,10))
#		S.add(db.StaticText('test\n1\n2'))
#		S.add(db.Bar('y'))
#
#		dashboard.draw_all()
#
#		screen.nodelay(True)
#
#		for Y in range(100):
#
#			c=screen.getch()
#
#			screen.addstr(20,0,str(c))
#
#			session.remember('y',Y/100)
#			C.draw()
#			time.sleep(1)
#
#
#	cs.wrapper(wrapped)