import dashboard as db
import pdb
import config as cfg
import numpy as np
import sys
from config import session


import curses as cs




class CDisplay:
	def __init__(self,display,xlim,ylim):
		self.display=display
		self.xlim=xlim
		self.ylim=ylim
		self.pad=cs.newpad(400,400)	
		display.concretization=self


	def draw(self):
		self.pad.erase()
		x1,x2=self.xlim
		y1,y2=self.ylim
		lines=self.display.getlines()
		for i,line in enumerate(lines):
			self.pad.addstr(i,0,line)

		self.pad.refresh(0,0,y1,x1,y2,x2)

	def poke(self,src):
		self.draw()




class CDashboard(db.AbstractDashboard):

	Concretedisplay=CDisplay
	def __init__(self,w,h):
		super().__init__()
		self.width=w
		self.height=h
		cfg.rawlogprint=False


instr='\n\nPress l (lowercase L) to generate learning plots.\nPress f to generate functions plot.\nPress q to quit.'
session.trackcurrent('statusinfo',instr)

def getwrapped(runfn,process_input=cfg.donothing):

	def got_input(c):
		if c==113: quit()
		else: process_input(c)
		cs.flushinp()

	def wrapped(screen):

		cs.use_default_colors()
		h=cs.LINES
		w=cs.COLS
		cfg.dashboard=CDashboard(w,h)

		blank=cs.newpad(400,400)
		blank.refresh(0,0,0,0,h-1,w-1)
		screen.refresh()

		#dummywin=cs.newwin(0,0,0,0)
		#dummywin.nodelay(True)

		screen.nodelay(True)

		def poke(*args,**kw):
			got_input(screen.getch())
			cfg.dashboard.draw_all()
			pass

		cfg.checkforinput=poke

		runfn()
	return wrapped






def RID(runfn,*x,**y):
	wrapped=getwrapped(runfn,*x,**y)
	cs.wrapper(wrapped)




# test
#
if __name__=='__main__':

	import time

	def wrapped(screen):
		#cs.use_default_colors()
		h=cs.LINES
		w=cs.COLS

		dashboard=CDashboard(50,20)

		S=db.StackedDisplay(memory=session,width=w)
		C=dashboard.add(S,(0,w-1),(0,10))
		S.add(db.StaticText('test\n1\n2'))
		S.add(db.Bar('y'))

		dashboard.draw_all()

		screen.nodelay(True)

		for Y in range(100):

			c=screen.getch()

			screen.addstr(20,0,str(c))

			session.remember('y',Y/100)
			C.draw()
			time.sleep(1)


	cs.wrapper(wrapped)