import dashboard as db
import config as cfg
import numpy as np
import sys


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------


class Slate_curses(db.AbstractSlate):

	def __init__(self,*args,**kw):
		super().__init__(*args)
		self.pad=kw['pad']
		x1,x2,y1,y2=kw['coords']
		self.x1=x1
		self.x2=x2
		self.y1=y1
		self.y2=y2
		self.shuffledcoords=[y1,x1,y2,x2]

	def cols(self):
		return self.x2-self.x1

	def draw(self):
		for ln,element in self.elements:
			self.pad.addstr(ln,0,element.getstr_safe())
		self.pad.refresh(0,0,*self.shuffledcoords)


def run_as_cs(function,*args,**kwargs):

	import curses
	import curses as cs
	

#----------------------------------------------------------------------------------------------------
	def temp(screen):

		cs.use_default_colors()

		h=curses.LINES
		w=curses.COLS
		p=cs.newpad(500,500)
		info=Slate_curses('log','refresh',pad=curses.newpad(500,500),coords=(0,w-1,0,10))
		log=Slate_curses('log','refresh',pad=curses.newpad(500,500),coords=(0,w-1,11,h-14))
		bars=Slate_curses('log','refresh',pad=curses.newpad(500,500),coords=(0,w-1,h-10,h-1))

		info.addtext(lambda *_:cfg.getval('sessioninfo'),height=10)

		log.addline()
		log.addtext(lambda *_:[s for s in cfg.gethist('log')[-(h-30):]],height=h-30)
		log.addline()

		bars.addtext('training loss of 10, 100 minibatches')
		bars.addbar(lambda *_:np.average(np.array(cfg.gethist('minibatch loss'))[-10:]),emptystyle='.')
		bars.addbar(lambda *_:np.average(np.array(cfg.gethist('minibatch loss'))[-100:]),emptystyle='.')
		bars.addspace(2)
		bars.addtext(lambda *_:'test loss {:.2}'.format(cfg.getval('test loss')))
		bars.addbar(lambda *_:cfg.getval('test loss'),emptystyle='.')
	#----------------------------------------------------------------------------------------------------
		function(*args,**kwargs)
		
	cs.wrapper(temp)

	

"""
# def run_as_cs_test(function,*args,**kwargs):
# 
# 	import curses as cs
# 	#----------------------------------------------------------------------------------------------------
# 	def prepdash():
# 
# 		s=Slate_curses('hello',pad=cs.newpad(500,500),coords=(0,0,50,50))
# 		s.addtext('x')
# 		s.addbar(lambda *_:cfg.getval('x')/100)
# 		s.addtext('y')
# 		s.addbar(lambda *_:1-cfg.getval('y')/100)
# 
# 	#----------------------------------------------------------------------------------------------------
# 
# 	def temp(screen):
# 		prepdash()
# 		function(*args,**kwargs)
# 	
# 	cs.wrapper(temp)
# 
# def runtest():
# 	run_as_cs_test(db.test,100)
"""



if __name__=='__main__':

	import e1,e2
	e={'e1':e1,'e2':e2}[sys.argv[1]]
	run_as_cs(e.run,sys.argv[2:])




