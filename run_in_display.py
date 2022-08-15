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

#----------------------------------------------------------------------------------------------------
def getwrapped(run,on_pause):

	def wrapped(screen):


		cs.use_default_colors()
		h=cs.LINES
		w=cs.COLS

		infodisplay,statusdisplay,logdisplay,dbprintdisplay=db.get4displays(w)
		dashboard=CDashboard(w)

		dashboard.add_display(infodisplay,0,0)
		dashboard.add_display(statusdisplay,0,w//2,name='status')
		dashboard.add_display(logdisplay,25,0)
		dashboard.add_display(dbprintdisplay,25,w//2)

		statuswindow=dashboard.displays['status'][1]
		cfg.dashboard=dashboard




		def setrunning():
			cs.flushinp()
			screen.nodelay(True)
			statuswindow.nodelay(True)
			session.trackcurrent('statusinfo','running, press p to pause or q to quit')

		def setpaused():
			cs.flushinp()
			screen.nodelay(False)
			statuswindow.nodelay(False)
			on_pause()

		def poke():
			got_input(statuswindow.getch())

		def got_input(c):
			if c==112:
				setpaused()
				while True:
					if statuswindow.getch()==99:
						setrunning()
						break
			if c==113:
				quit()

		cfg.poke=poke
		setrunning()
		run()
	return wrapped


def on_pause():
	session.trackcurrent('statusinfo','paused, press c to continue')


def RID(run,on_pause=on_pause):
	wrapped=getwrapped(run,on_pause)
	cs.wrapper(wrapped)

