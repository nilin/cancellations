import sys
import os
import math
import numpy as np
import pdb
import config as cfg
import time


#----------------------------------------------------------------------------------------------------


BOX='\u2588'
box='\u2592'
dash='\u2015'

def clear():
	os.system('cls' if os.name == 'nt' else 'clear')

#----------------------------------------------------------------------------------------------------
# display tracked values
#----------------------------------------------------------------------------------------------------




class AbstractSlate():

	def __init__(self,*signals):
		self.elements=[]
		self.ln=1
		cfg.addlistener(self,*signals)
		self.signals=list(signals)+['errlog']
		self.trackedvarnames=set()
		self.memory=cfg.State()

	def trackvars(self,*names):
		self.trackedvarnames.update(names)
		cfg.addlistener(self,*names)

	def add(self,display,height=1):
		self.elements.append((self.ln,display))
		self.ln=self.ln+height

	def poke(self,signal,*args):
		if signal in self.signals:
			self.refresh()
		if signal in self.trackedvarnames:
			name=signal
			self.memory.remember(name,cfg.getval(name))

	def refresh(self):
		self.draw()
		#print('debug prints: {}'.format(cfg.dbprintbuffer[-1]))

		

	def addbar(self,fn,**kwargs):
		self.add(Bar(fn,self,**kwargs))

	def addline(self,style=dash):
		self.add(Text(style,self,emptystyle=style))

	def addtext(self,msg,height=1):
		self.add(Text(msg,self,height=height),height)

	def addspace(self,n=1):
		self.ln=self.ln+n

	@staticmethod
	def cols():
		return os.get_terminal_size()[0]-1

	
		
class DisplayElement:
	def __init__(self,fn,slate,height=1,emptystyle=' ',**kwargs):
		self.fn=fn
		self.slate=slate
		self.kwargs=kwargs
		self.height=height
		self.Emptystyle=math.ceil(self.slate.cols()/len(emptystyle))*emptystyle

	def getstr_safe(self):
		try:
			s=self.getstr()
		except Exception as e:
			s='pending...{}'.format(str(e))
		return '\n'.join([l+self.Emptystyle[-(self.slate.cols()-len(l)):] for l in s.splitlines()[-self.height:]])

class Bar(DisplayElement):
	def __init__(self,*x,style=cfg.BOX,**y):
		#if 'emptystyle' not in y:y['emptystyle']='.'
		super().__init__(*x,**y)
		self.Style=math.ceil(self.slate.cols()/len(style))*style

	def getstr(self):
		val=self.fn(self.slate.memory)
		return barstring(val,self.slate.cols(),Style=self.Style)
	
class Text(DisplayElement):
	def getstr(self):
		if type(self.fn)==str:
			return self.fn
		else:
			msg=self.fn(self.slate.memory)
		return '\n'.join(msg) if type(msg)==list else msg


def barstring(val,fullwidth,Style):
	barwidth=math.floor(fullwidth*min(val,1))
	return Style[:barwidth]+cfg.BOX


def wideline():
	return (os.get_terminal_size()[0]-1)*dash
	




#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------



class Slate(AbstractSlate):

	def __init__(self,*args):
		super().__init__(*args)
		clear()


	@staticmethod
	def gotoline(n):
		print('\x1b['+str(n)+';0H')

	def draw(self):
		for ln,element in self.elements:
			self.gotoline(ln)
			print(element.getstr_safe())




# display on a single line (less likely to mess up terminal)

class MinimalSlate(AbstractSlate):

	def __init__(self,*args):
		super().__init__(*args)
		print(5*'\n')

	def cols(self):
		return self.elementwidth()

	def elementwidth(self):
		return os.get_terminal_size()[0]//max(len(self.elements),1)-3

	def draw(self):
		print((' {} '.format(cfg.box)).join([element.getstr_safe().replace('\n','|')[:self.elementwidth()] for _,element in self.elements]),end='\r')








#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------



def display_0():
	slate=MinimalSlate('refresh','log','block')
	slate.addtext(lambda *_:[s for s in cfg.getrecentlog(1)],height=1)
	slate.addbar(lambda *_:np.average(np.array(slate.gethist('minibatch loss')[1])[10:]),style='training loss ',emptystyle='.')
	return slate

def display_1(*_):

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
	slate.addtext(lambda *_:'\n'.join([str(msg) for msg in cfg.dbprintbuffer[-5:]]),height=10)
	#slate.addline()
	#slate.addbar(lambda *_:cfg.getval('test loss'),emptystyle='.')
	return slate


def display_2(*_):

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
	#slate.addbar(lambda memory,*_:np.average(memory.gethist('minibatch loss')[1][-10:]))
	slate.addtext(lambda memory,*_:'{:.2f}'.format(np.average(memory.gethist('minibatch loss')[1][-100:])))
	#slate.addbar(lambda memory,*_:np.average(memory.gethist('minibatch loss')[1][-100:]))
	slate.addspace(2)
	#slate.addtext(lambda memory,*_:'test loss {:.2}'.format(np.average(memory.gethist('quick test loss')[1][-10])))
	#slate.addbar(lambda memory,*_:np.average(memory.gethist('quick test loss')[1][-10]))

	slate.addtext(lambda *_:'epoch {}% done'.format(int(100*(1-cfg.getval('minibatches left')/cfg.getval('minibatches')))))
	slate.addspace(1)
	slate.addbar(lambda *_:cfg.getval('block')[0]/cfg.getval('block')[1],style='sample blocks done ')

	slate.addspace(2)
	slate.addline()
	slate.addtext('debug prints (cfg.dbprint(msg))')
	slate.addtext(lambda *_:'\n'.join([str(msg) for msg in cfg.dbprintbuffer[-5:]]),height=10)
	#slate.addline()
	#slate.addbar(lambda *_:cfg.getval('test loss'),emptystyle='.')
	return slate







#====================================================================================================
# testing
#====================================================================================================



def prepdash(s):

	s.addtext('x')
	s.addbar(lambda *_:cfg.getval('x')/100)
	s.addtext('y')
	s.addbar(lambda *_:1-cfg.getval('y')/100)


def test(n):
	for Y in range(n):
		cfg.trackcurrent('y',Y)
		for X in range(n):
			cfg.trackcurrent('x',X)
			cfg.pokelisteners('hello')
			time.sleep(.001)


if __name__=='__main__':

	print('\nrunning test of dashboard\n')


	#s=Slate('hello')
	s=MinimalSlate('hello')
	
	prepdash(s)
	test(100)	





