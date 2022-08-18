import dashboard as db
import pdb
import config as cfg
import numpy as np
import sys
from config import session


import curses as cs


class CDashboard(db.AbstractDashboard):

	def __init__(self,w):
		super().__init__()
		self.width=w

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

paused=False

instr='\n\nPress l (lowercase L) to generate learning plots.\nPress f to generate functions plot.\nPress q to quit.'

def getwrapped(run,process_input=cfg.donothing):

	def wrapped(screen):


		cs.use_default_colors()
		h=cs.LINES
		w=cs.COLS

		infodisplay,statusdisplay,logdisplay,dbprintdisplay=db.get4displays(w)
		dashboard=CDashboard(w)

		dashboard.add_display(infodisplay,0,0)
		dashboard.add_display(statusdisplay,0,w//2,name='status')
		dashboard.add_display(logdisplay,20,0)
		dashboard.add_display(dbprintdisplay,20,w//2)

		statuswindow=dashboard.displays['status'][1]
		cfg.dashboard=dashboard



		def setrunning():
			global paused
			paused=False
			cs.flushinp()
			#screen.nodelay(True)
			statuswindow.nodelay(True)
			session.trackcurrent('statusinfo','running, press p to pause'+instr)

		def setpaused():
			cs.flushinp()
			global paused
			paused=True
			#screen.nodelay(False)
			statuswindow.nodelay(False)
			session.trackcurrent('statusinfo','paused, press c to continue'+instr)
			while paused:
				got_input(statuswindow.getch())

		def poke():
			got_input(statuswindow.getch())

		def got_input(c):
			if c==112:
				setpaused()
			if c==99:
				setrunning()
			if c==113:
				quit()
			process_input(c)

		cfg.poke=poke
		setrunning()
		run()
	return wrapped


def RID(run,*x,**y):
	wrapped=getwrapped(run,*x,**y)
	cs.wrapper(wrapped)

