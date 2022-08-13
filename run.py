import dashboard as db
import config as cfg
import numpy as np
import sys

from exRatio import run



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
def temp(screen):

	cs.use_default_colors()
	h=cs.LINES
	w=cs.COLS

	infodisplay,logdisplay,dbprintdisplay=db.get3displays(w)
	dashboard=CDashboard(w)

	dashboard.add_display(infodisplay,0,0)
	dashboard.add_display(logdisplay,25,0)
	dashboard.add_display(dbprintdisplay,25,w//2)

	dashboard.draw_all()
	cfg.dashboard=dashboard

	run()

cs.wrapper(temp)

	


#if __name__=='__main__':

	#import e1,test
	#e=sys.argv[1]
	#run_as_cs({'ex':e1,'test':test}[e].run)






