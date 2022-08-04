import sys
import os
import math
import numpy as np
import jax.numpy as jnp
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

	def _addbar_(self,fn,**kwargs):
		self.add(Bar(fn,self,**kwargs))

	@staticmethod
	def varval(var,transform=lambda x:x,avg_of=1,**kwargs):
		return lambda memory,*_:np.average(transform(jnp.array(memory.gethist(var)[1])[-avg_of:]))

	def addbar(self,var,**kwargs):
		fn=AbstractSlate.varval(var,**kwargs)
		self._addbar_(fn,**kwargs)

	def addline(self,style=dash):
		self.add(Text(style,self,emptystyle=style))

	def addtext(self,msg,height=1,**kwargs):
		self.add(Text(msg,self,height=height),height)

	def addvarprint(self,var,formatting=lambda x:x,**kwargs):
		fn=AbstractSlate.varval(var,**kwargs)
		self.addtext(lambda *x:str(formatting(fn(*x))))
		#self.addtext(fn,**kwargs)

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



class Display0(Slate):

	def __init__(self,*args):
		super().__init__('refresh','log','block')
		self.addtext(lambda *_:cfg.getval('sessioninfo'),height=15)
		self.addline()
		self.addtext('log')
		self.addtext(lambda *_:cfg.getrecentlog(10),height=10)
		self.addline()
		self.addtext('prints (cfg.dbprint(msg))')
		self.addtext(lambda *_:'\n'.join([line for msg in cfg.dbprintbuffer[-10:] for line in str(msg).split('\n')][-10:]),height=10)
		self.addline()
		self.addspace(2)


class Display1(Display0):

	def __init__(self,*args):
		super().__init__()
		self.trackvars('minibatch loss','quick test loss')
		self.addtext('training loss of 10, 100 minibatches')
		self.addvarprint('minibatch loss',formatting=lambda x:'{:.2f}'.format(x),avg_of=10)
		self.addbar('minibatch loss',avg_of=10)
		self.addvarprint('minibatch loss',formatting=lambda x:'{:.2f}'.format(x),avg_of=100)
		self.addbar('minibatch loss',avg_of=100)
		self.addspace(2)

		self.addtext(lambda *_:'epoch {}% done'.format(int(100*(1-cfg.getval('minibatches left')/cfg.getval('minibatches')))))
		self.addspace(1)
		self.addbar(lambda *_:cfg.getval('block')[0]/cfg.getval('block')[1],style='sample blocks done ')



#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------



def mindisplay():
	slate=MinimalSlate('refresh','log','block')
	slate.addtext(lambda *_:[s for s in cfg.getrecentlog(1)],height=1)
	slate.addbar(lambda *_:np.average(np.array(slate.gethist('minibatch loss')[1])[10:]),style='training loss ',emptystyle='.')
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





