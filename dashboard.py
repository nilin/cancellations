import sys
import os
import math
import numpy as np
import jax.numpy as jnp
import pdb
import config as cfg
import time
from config import session


#----------------------------------------------------------------------------------------------------


BOX='\u2588'
box='\u2592'
dash='\u2015'

def clear():
	os.system('cls' if os.name == 'nt' else 'clear')




#----------------------------------------------------------------------------------------------------



class Display:
	def __init__(self,height,width,memory,emptystyle=' ',**kwargs):
		self.height=height
		self.width=width
		self.Emptystyle=math.ceil(self.width/len(emptystyle))*emptystyle
		self.memory=memory

	def getcroppedlines(self):
		try:
			lines=self.getlines()[:self.height]
		except:
			lines=['pending']

		lines=[line[:self.width] for line in lines]
		return [line+self.Emptystyle[-(self.width-len(line)):] for line in lines]

	def getcroppenstr(self):
		return '\n'.join(self.getcroppedlines())


class Text(Display):
	def __init__(self,height,width,memory,query,formatting='{}',**kwargs):
		super().__init__(height,width,memory,**kwargs)
		self.query=query
		self.formatting=formatting

	def getlines(self):
		return self.formatting.format(memory.getcurrentval(self.query)).splitlines()
	

class StaticText(Display):
	def __init__(self,width,text,**kwargs):
		self.text=text.splitlines()
		height=len(self.text)
		super.__init__(height,width,None,**kwargs)

	def getlines(self):
		return self.text


class NumberDisplay(Display):
	def __init__(self,width,memory,query,transform=None,avg_of=1,**kwargs):
		super().__init__(1,width,memory,query,**kwargs)
		self.hist=cfg.History()
		self.query=query
		self.transform=(lambda x:x) if transform==None else transform
		self.avg_of=avg_of

	def getlines(self):
		out=self.transform(self.memory.getcurrentval(self.query))
		self.hist.remember(out)
		#value_shown=jnp.average(jnp.array(self.hist.gethist())[-self.avg_of:])
		value_shown=self.hist.getcurrentval() #jnp.average(jnp.array(self.hist.gethist())[-self.avg_of:])
		return [self.formatnumber(value_shown)]



class Bar(NumberDisplay):
	def __init__(self,*x,style=cfg.BOX,**y):
		super().__init__(*x,**y)
		self.Style=math.ceil(self.width/len(style))*style

	def formatnumber(self,x):
		barwidth=math.floor(self.width*min(x,1))
		return self.Style[:barwidth]+cfg.BOX



class NumberPrint(NumberDisplay):
	def __init__(self,*x,msg='{:.3}',**y):
		super().__init__(*x,**y)
		self.msg=msg

	def formatnumber(self,x):
		return self.msg.format(x)



class StackedDisplay(Display):
	
	def __init__(self,*x,**y):
		super().__init__(*x,**y)
		self.displays=[]

	def getlines(self):
		return [line for display in self.displays for line in display.getlines()]

	def add(self,display):
		self.displays.append(display)

	def addbar(self,query,**kwargs):
		self.add(Bar(self.width,self.memory,query,**kwargs))

	def addnumberprint(self,query,**kwargs):
		self.add(NumberPrint(self.width,self.memory,query,**kwargs))

	def addtext(self,height,query,**kwargs):
		self.add(Text(height,self.width,self.memory,query))

	def addstatictext(self,text,**kwargs):
		self.add(StaticText(self.width,text,**kwargs))

	def addline(self):
		self.addstatictext('',emptystyle=dash)
	

#----------------------------------------------------------------------------------------------------


class Display0(StackedDisplay):
	
	def __init__(self,height,width):
		super().__init__(height,width,session)

		self.addline()
		self.addstatictext('session info')
		self.addline()
		self.addtext(15,'sessioninfo')

		self.addline()
		self.addstatictext('log')
		self.addline()
		self.addtext(15,'log')

		self.addline()
		self.addstatictext('prints (cfg.print(msg))')
		self.addline()
		self.addtext(15,'dbprintbuffer')

		




#----------------------------------------------------------------------------------------------------




class Slate():
	def __init__(self):
		self.displays=dict()

	def add_display(self,name,display,y,x=0):
		self.displays[name]=(display,y,x)
		cfg.addlistener(self,name)

	@staticmethod
	def print_at(y,x,msg):
		print('\x1b[{};{}H{}'.format(y,x,msg))

	def draw(self,name):
		display,y,x=self.displays[name]
		lines=display.getcroppedlines()
		for i,line in enumerate(lines):
			Slate.print_at(y+i,x,line)

	def poke(self,name):
		self.draw(name)


#
#
#
#class Display0(Slate):
#
#	def __init__(self,*args):
#		super().__init__(memory=session)
#		self.addtext(['sessioninfo'],height=15)
#		self.addline()
#		self.addstatictext('log')
#		self.addtext(['log'],transform=lambda x:x[0][-10:],height=10)
#		self.addline()
#		self.addstatictext('prints (cfg.dbprint(msg))')
#		#self.addtext(lambda *_:'\n'.join([line for msg in cfg.dbprintbuffer[-10:] for line in str(msg).split('\n')][-10:]),height=10)
#		self.addline()
#		self.addspace(2)
#
#
#class Display1(Display0):
#
#	def __init__(self,*args):
#		super().__init__()
#		self.trackvars('minibatch loss','quick test loss')
#		self.addtext('training loss of 10, 100 minibatches')
#		self.addvarprint('minibatch loss',formatting=lambda x:'{:.2f}'.format(x),avg_of=10)
#		self.addbar('minibatch loss',avg_of=10)
#		self.addvarprint('minibatch loss',formatting=lambda x:'{:.2f}'.format(x),avg_of=100)
#		self.addbar('minibatch loss',avg_of=100)
#		self.addspace(2)
#
#		self.addtext(lambda *_:'epoch {}% done'.format(int(100*(1-cfg.getval('minibatches left')/cfg.getval('minibatches')))))
#		self.addspace(1)
#		self.addbar(lambda *_:cfg.getval('block')[0]/cfg.getval('block')[1],style='sample blocks done ')
#
#

#class AbstractSlate(cfg.Listener):
#
#	def __init__(self,memory=None):
#		self.elements=[]
#		self.ln=1
#		self.signals=[]
#		self.trackedvarnames=set()
#		self.memory=cfg.Memory() if memory==None else memory
#
#	def listen(self,*signals):
#		cfg.addlistener(self,*signals)
#		self.signals=self.signals+signals
#
#	def trackvars(self,*names):
#		self.trackedvarnames.update(names)
#		cfg.addlistener(self,*names)
#
#	def add(self,display,height=1):
#		self.elements.append((self.ln,display))
#		self.ln=self.ln+height
#
#	def refresh(self):
#		self.draw()
#
#	# active displays -------------------------------------------------------------
#
#	def addbar(self,queries,transform=None,**kwargs):
#		self.add(Bar(queries,transform,self,**kwargs))
#
#	def addtext(self,queries,transform=None,height=1,**kwargs):
#		self.add(Text(queries,transform,self,height=height),height)
#
#
#
#	# static displays -------------------------------------------------------------
#
#	def addstatictext(self,msg,**kwargs):
#		self.addtext(None,lambda *_:msg,**kwargs)
#
#	def addline(self,style=dash):
#		self.addstatictext('',emptystyle=style)
#
#	def addspace(self,n=1):
#		self.ln=self.ln+n
#
#	@staticmethod
#	def cols():
#		return os.get_terminal_size()[0]-1

#class AbstractSlate(cfg.Listener):
#
#	def __init__(self,memory=None):
#		self.elements=[]
#		self.ln=1
#		self.signals=[]
#		self.trackedvarnames=set()
#		self.memory=cfg.Memory() if memory==None else memory
#
#	def listen(self,*signals):
#		cfg.addlistener(self,*signals)
#		self.signals=self.signals+signals
#
#	def trackvars(self,*names):
#		self.trackedvarnames.update(names)
#		cfg.addlistener(self,*names)
#
#	def add(self,display,height=1):
#		self.elements.append((self.ln,display))
#		self.ln=self.ln+height
#
#	def refresh(self):
#		self.draw()
#
#	# active displays -------------------------------------------------------------
#
#	def addbar(self,queries,transform=None,**kwargs):
#		self.add(Bar(queries,transform,self,**kwargs))
#
#	def addtext(self,queries,transform=None,height=1,**kwargs):
#		self.add(Text(queries,transform,self,height=height),height)
#
#
#
#	# static displays -------------------------------------------------------------
#
#	def addstatictext(self,msg,**kwargs):
#		self.addtext(None,lambda *_:msg,**kwargs)
#
#	def addline(self,style=dash):
#		self.addstatictext('',emptystyle=style)
#
#	def addspace(self,n=1):
#		self.ln=self.ln+n
#
#	@staticmethod
#	def cols():
#		return os.get_terminal_size()[0]-1
#

#class Slate(AbstractSlate):
#	def __init__(self,*args,**kwargs):
#		super().__init__(*args,**kwargs)
#		clear()
#
#
#	@staticmethod
#	def gotoline(n):
#		print('\x1b['+str(n)+';0H')
#
#
#	@staticmethod
#	def goto(y,x):
#		print('\x1b['+str(y)+';'+str(x)+'H')
#
#
#	def draw(self):
#		for ln,element in self.elements:
#			self.gotoline(ln)
#			print(element.getstr_safe())
#






#====================================================================================================
# testing
#====================================================================================================




def test(n):


	for Y in range(n):
		session.remember('y',Y)
		for X in range(n):
			session.remember('x',X)


			cfg.pokelisteners('hello')
			time.sleep(.001)


if __name__=='__main__':

	print('\nrunning test of dashboard\n')

	
	d=StackedDisplay(10,50,session)
	d.addbar('y',transform=lambda x:x/100)
	d.addbar('x',transform=lambda x:x/100)
	d.addnumberprint('x',transform=lambda x:x/100)

	s=Slate()
	s.add_display('hello',d,10,10)

	test(100)



