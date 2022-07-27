import example
import os
import pdb
import dashboard as db
import curses as cs
import config as cfg
import numpy as np






class CSlate(db.AbstractSlate):

	def put_on_pad(self,pad,*coords):
		self.pad=pad
		self.coords=coords

	def cols(self):
		return self.coords[-1]-self.coords[-3]

	def draw(self):
		for ln,element in self.elements:
			self.pad.addstr(ln,0,element.getstr_safe())
		self.pad.refresh(0,0,*self.coords)



info=CSlate()
log=CSlate()
bars=CSlate()

info.addtext(lambda tk:tk.get('sessioninfo'),height=100)
log.addtext(lambda tk:list(reversed(tk.gethist('log')))[:100],height=100)

bars.addtext('training loss of last 10, 100 minibatches')
bars.addbar(lambda tk:np.average(np.array(tk.gethist('minibatch loss'))[-10:]),emptystyle='.')
bars.addbar(lambda tk:np.average(np.array(tk.gethist('minibatch loss'))[-100:]),emptystyle='.')
bars.addspace()
bars.addtext('validation loss')
bars.addbar(lambda tk:tk.get('validation loss'),emptystyle='.')




def run(screen):

	h=cs.LINES
	w=cs.COLS
	p1=cs.newpad(500,500)
	p2=cs.newpad(500,500)
	p3=cs.newpad(500,500)

	log.put_on_pad(p1,0,0,h-1,49)
	info.put_on_pad(p2,0,50,20,100)
	bars.put_on_pad(p3,20,50,40,w-1)

	example.run()

#cs.wrapper(run)

screen=cs.initscr()
try:
	run(screen)
except:
	pass
cs.endwin()
