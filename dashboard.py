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

def print_at(y,x,msg):
	print('\x1b[{};{}H{}'.format(y,x,msg))


#----------------------------------------------------------------------------------------------------



class Display:
	def __init__(self,height,width,memory,emptystyle=' ',**kwargs):
		self.height=height
		self.width=width
		self.Emptystyle=math.ceil(self.width/len(emptystyle))*emptystyle
		self.memory=memory
		self.trackedvars=[]

	def getlines_wrapped(self):
		try:
			return self.getlines()
		except Exception as e:
			return ['pending '+str(e)]

	def getcroppedlines(self):
		lines=self.getlines()[:self.height]

		#lines=[line.splitlines()[0] for line in lines]
		lines=[line[:self.width] for line in lines]

		#pdb.set_trace()

		return [line+self.Emptystyle[-(self.width-len(line)):] for line in lines]

	def getcroppedstr(self):
		return '\n'.join(self.getcroppedlines())



class StackedDisplay(Display):
	
	def __init__(self,*x,**y):
		super().__init__(*x,**y)
		self.displays=[]

	def getlines(self):
		return [line for display in self.displays for line in display.getlines_wrapped()]

	def add(self,display):
		self.displays.append(display)
		self.trackedvars=self.trackedvars+display.trackedvars

	def addbar(self,query,**kwargs):
		self.add(Bar(self.width,self.memory,query,**kwargs))

	def addnumberprint(self,query,**kwargs):
		self.add(NumberPrint(self.width,self.memory,query,**kwargs))

	def addhistdisplay(self,height,query,**kwargs):
		self.add(HistDisplay(height,self.width,self.memory,query))

	def addstatictext(self,text,**kwargs):
		self.add(StaticText(self.width,text,**kwargs))

	def addline(self):
		self.addstatictext(dash*self.width)

	def addspace(self,n=1):	
		for i in range(n):
			self.addstatictext(' '*self.width)


class HistDisplay(Display):
	def __init__(self,height,width,memory,query,formatting=None,**kwargs):
		super().__init__(height,width,memory,**kwargs)
		self.query=query
		self.formatting=lambda lines:lines[-height:] if formatting==None else formatting
		self.trackedvars=[query]

	def getlines(self):
		return self.formatting(self.memory.gethist(self.query))
	

class StaticText(Display):
	def __init__(self,width,text,**kwargs):
		self.lines=text.splitlines()
		height=len(self.lines)
		super().__init__(height,width,None,**kwargs)

	def getlines(self):
		return self.lines


class NumberDisplay(Display):
	def __init__(self,width,memory,query,transform=None,avg_of=1,**kwargs):
		super().__init__(1,width,memory,query,**kwargs)
		self.hist=cfg.History()
		self.query=query
		self.transform=(lambda x:x) if transform==None else transform
		self.avg_of=avg_of
		self.trackedvars=[query]

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


#----------------------------------------------------------------------------------------------------


class Display0(StackedDisplay):
	
	def __init__(self,height,width):
		super().__init__(height,width,session)

		#self.addstatictext('session info')
		#self.addline()
		self.addstatictext(self.memory.getcurrentval('sessioninfo'))
		self.addspace(2)

		self.addstatictext('log')
		self.addline()
		self.addhistdisplay(15,'log')
		self.addspace(2)

		self.addstatictext('prints (cfg.print(msg))')
		self.addline()
		self.addhistdisplay(15,'dbprintbuffer')





#----------------------------------------------------------------------------------------------------




class Dashboard:

	def __init__(self):
		self.displays=[]
		clear()

	def add_display(self,display,y,x=0):
		self.displays.append((display,y,x))
		display.memory.addlistener(self)

	def poke(self,signal):
		self.draw(signal)





class Dashboard0(Dashboard):

	def __init__(self):
		super().__init__()
		clear()
		self.width=os.get_terminal_size()[0]-1

	def draw(self,signal=None):
		for (display,y,x) in self.displays:
			if signal in display.trackedvars or signal==None:
				lines=display.getcroppedlines()

				for i,line in enumerate(lines):
					print_at(y+i+1,x,line)
	





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

	db=Dashboard0()
	db.add_display('hello',d,10,10)

	test(100)



