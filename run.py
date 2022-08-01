import dashboard as db
import config as cfg
import numpy as np
import sys


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------


class Slate_curses(db.AbstractSlate):

	def __init__(self,*args,**kw):
		super().__init__(*args)
		self.trackvars('minibatch loss','quick test loss')

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
		info=Slate_curses('log','refresh','block',pad=curses.newpad(500,500),coords=(0,w-1,0,19))
		log=Slate_curses('log','refresh','block',pad=curses.newpad(500,500),coords=(0,w-1,20,29))
		debug=Slate_curses('log','refresh','block',pad=curses.newpad(500,500),coords=(0,w-1,30,39))
		bars=Slate_curses('log','refresh','block',pad=curses.newpad(500,500),coords=(0,w-1,40,h-1))

		info.addtext(lambda *_:cfg.getval('sessioninfo'),height=18)

		log.addline()
		log.addtext('log')
		log.addtext(lambda *_:cfg.getrecentlog(8),height=8)
		log.addline()

		debug.addtext('debug prints (cfg.dbprint(msg))')
		debug.addtext(lambda *_:str(cfg.dbprintbuffer[-1]),height=8)

		bars.addtext('training loss of 10, 100 minibatches')
		bars.addtext(lambda memory,*_:'{:.2f}'.format(np.average(memory.gethist('minibatch loss')[1][-10:])))
		bars.addbar(lambda memory,*_:np.average(memory.gethist('minibatch loss')[1][-10:]))
		bars.addtext(lambda memory,*_:'{:.2f}'.format(np.average(memory.gethist('minibatch loss')[1][-100:])))
		bars.addbar(lambda memory,*_:np.average(memory.gethist('minibatch loss')[1][-100:]))
		bars.addspace(2)

		bars.addtext(lambda *_:'epoch {}% done'.format(int(100*(1-cfg.getval('minibatches left')/cfg.getval('minibatches')))))
		bars.addspace(1)
		bars.addbar(lambda *_:cfg.getval('block')[0]/cfg.getval('block')[1],style='sample blocks done ')

		cfg.dblog('in temp 2')
	#----------------------------------------------------------------------------------------------------
		function(*args,**kwargs)
		
	cs.wrapper(temp)

	



"""

	slate=Slate('refresh','log','block')
	slate.trackvars('minibatch loss','quick test loss')
	slate.addtext(lambda *_:cfg.getval('sessioninfo'),height=15)
	slate.addline()
	slate.addtext('log')
	slate.addtext(lambda *_:cfg.getrecentlog(20),height=20)
	slate.addline()
	slate.addspace(2)
	slate.addtext('training loss of 10, 100 minibatches')
	slate.addtext(lambda memory,*_:'{:.2f}'.format(np.average(memory.gethist('minibatch loss')[1][-10:])))
	slate.addbar(lambda memory,*_:np.average(memory.gethist('minibatch loss')[1][-10:]))
	slate.addtext(lambda memory,*_:'{:.2f}'.format(np.average(memory.gethist('minibatch loss')[1][-100:])))
	slate.addbar(lambda memory,*_:np.average(memory.gethist('minibatch loss')[1][-100:]))
	slate.addspace(2)
	#slate.addtext(lambda memory,*_:'test loss {:.2}'.format(np.average(memory.gethist('quick test loss')[1][-10])))
	#slate.addbar(lambda memory,*_:np.average(memory.gethist('quick test loss')[1][-10]))

	slate.addtext(lambda *_:'epoch {}% done'.format(int(100*(1-cfg.getval('minibatches left')/cfg.getval('minibatches')))))
	slate.addspace(1)
	slate.addbar(lambda *_:cfg.getval('block')[0]/cfg.getval('block')[1],style='sample blocks done ')

	slate.addspace(2)
	slate.addline()
	slate.addtext('debug prints (cfg.dbprint(msg))')
	slate.addtext(lambda *_:str(cfg.dbprintbuffer[-1]),height=10)
	#slate.addline()
	#slate.addbar(lambda *_:cfg.getval('test loss'),emptystyle='.')

"""






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

	import e1,test
	e=sys.argv[1]
	run_as_cs({'e1':e1,'test':test}[e].run)




