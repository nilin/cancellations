import dashboard as db
import pdb
import config as cfg
import numpy as np
import sys
from config import session


import curses as cs


class CDashboard(db.AbstractDashboard):

	def __init__(self,w,h):
		super().__init__()
		self.width=w
		self.height=h

	def makeconcretedisplay(self,display,y,x):
		window=cs.newwin(display.height,display.width,y,x)	
		return (display,window)

	def getdisplay(self,concretedisplay):
		return concretedisplay[0]

	def draw(self,concretedisplay):
		display,window=concretedisplay
		lines=display.getcroppedlines()

		for i,line in enumerate(lines):
			window.addstr(i,0,line[:display.width-1])

		window.refresh()

#paused=False

instr='\n\nPress l (lowercase L) to generate learning plots.\nPress f to generate functions plot.\nPress q to quit.'
session.trackcurrent('statusinfo',instr)

def getwrapped(runfn,process_input=cfg.donothing):

	def wrapped(screen):

		cs.use_default_colors()
		h=cs.LINES
		w=cs.COLS

		screen.clear()

		dashboard=CDashboard(w,h)

		infodisplay,statusdisplay,logdisplay=db.get3displays(w,h)
		dashboard.add_display(infodisplay,0,0)
		dashboard.add_display(statusdisplay,0,w//2,name='status')
		dashboard.add_display(logdisplay,h//3,w//2)

		statuswindow=dashboard.displays['status'][1]
		cfg.dashboard=dashboard

		def poke(*args,**kw):
			got_input(statuswindow.getch())

		def got_input(c):
			if c==113: quit()
			else: process_input(c)

		cfg.poke=poke
		
		cs.flushinp()
		statuswindow.nodelay(True)

		runfn()
	return wrapped



def RID(runfn,*x,**y):
	wrapped=getwrapped(runfn,*x,**y)
	cs.wrapper(wrapped)

