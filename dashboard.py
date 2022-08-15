import sys
import os
import math
import numpy as np
import jax.numpy as jnp
import pdb
import config as cfg
from collections import deque
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



class Display(cfg.Watched,cfg.Stopwatch):
	def __init__(self,height,width,memory,emptystyle=' ',**kwargs):
		cfg.Watched.__init__(self)
		cfg.Stopwatch.__init__(self)

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

		lines=[line.splitlines()[0] if len(line)>0 else '' for line in lines]
		lines=[line[:self.width] for line in lines]

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


class QueriedText(Display):
	def __init__(self,height,width,memory,query):
		super().__init__(height,width,memory)
		self.query=query
		self.trackedvars=[query]

	def getlines(self):
		try:
			return self.memory.getval(self.query).splitlines()
		except:
			return ['pending']
		


class NumberDisplay(Display):
	def __init__(self,width,memory,query,transform=None,avg_of=1,**kwargs):
		super().__init__(1,width,memory,query,**kwargs)
		self.query=query
		self.transform=(lambda x:x) if transform==None else transform
		self.trackedvars=[query]

		if avg_of==1:
			self.getlines=self.getlines0
		else:
			self.hist=cfg.History()
			self.avg_of=avg_of
			self.getlines=self.getlines1

	def getlines0(self):
		out=self.transform(self.memory.getcurrentval(self.query))
		return [self.formatnumber(out)]

	def getlines1(self):
		out=self.transform(self.memory.getcurrentval(self.query))

		self.hist.remember(out)
		histvals=self.hist.gethist()
		histvals=histvals[-min(self.avg_of,len(histvals)):]
		return [self.formatnumber(sum(histvals)/len(histvals))]



class Bar(NumberDisplay):
	def __init__(self,*x,style=cfg.BOX,**y):
		super().__init__(*x,**y)
		self.Style=math.ceil(self.width/len(style))*style

	def formatnumber(self,x):
		barwidth=math.floor(self.width*max(min(x,1),0))
		return self.Style[:barwidth]#+cfg.BOX



class NumberPrint(NumberDisplay):
	def __init__(self,*x,msg='{:.3}',**y):
		super().__init__(*x,**y)
		self.msg=msg

	def formatnumber(self,x):
		return self.msg.format(x)





class AbstractDashboard:

	def __init__(self):
		clear()
		self.displays=dict()
		self.defaultnames=deque(range(100))

	def add_display(self,display,y,x=0,name=None):
		if name==None: name=self.defaultnames.popleft()
		display.addlistener(self)
		cd=self.makeconcretedisplay(display,y,x)
		self.displays[name]=cd
		display.memory.addlistener(self)

		self.draw(cd)
		return name

	def del_display(self,name):
		del self.displays[name]

	def poke(self,signal):
		for name,concretedisplay in self.displays.items():
			display=self.getdisplay(concretedisplay)
			if signal==None or signal in display.trackedvars:

				if display.tick()>.01:
					self.draw(concretedisplay)

	def draw_all(self):
		for name,concretedisplay in self.displays.items():
			self.draw(concretedisplay)


class Dashboard(AbstractDashboard):

	def makeconcretedisplay(self,display,y,x):
		return (display,y,x)

	def getdisplay(self,concretedisplay):
		return concretedisplay[0]

	def draw(self,concretedisplay):
		display,y,x=concretedisplay

		lines=display.getcroppedlines()
		for i,line in enumerate(lines):
			print_at(y+i+1,x,line[:display.width])


		

def get3displays(width):
		#width=os.get_terminal_size()[0]-1

	#infodisplay=StackedDisplay(25,width,session)
	#infodisplay.addhistdisplay(25,'sessioninfo')
	infodisplay=QueriedText(25,width,session,'sessioninfo')

	logdisplay=StackedDisplay(10,round(width*.4),session)
	logdisplay.addstatictext('log')
	logdisplay.addline()
	logdisplay.addhistdisplay(10,'log')

	dbprintdisplay=StackedDisplay(10,round(width*.4),session)
	dbprintdisplay.addstatictext('prints (cfg.dbprint(msg))')
	dbprintdisplay.addline()
	dbprintdisplay.addhistdisplay(10,'dbprintbuffer')

	return infodisplay,logdisplay,dbprintdisplay

def get4displays(width):
	infodisplay=QueriedText(25,round(width*.4),session,'sessioninfo')
	statusdisplay=QueriedText(25,round(width*.4),session,'statusinfo')

	logdisplay=StackedDisplay(10,round(width*.4),session)
	logdisplay.addstatictext('log')
	logdisplay.addline()
	logdisplay.addhistdisplay(10,'log')

	dbprintdisplay=StackedDisplay(10,round(width*.4),session)
	dbprintdisplay.addstatictext('prints (cfg.dbprint(msg))')
	dbprintdisplay.addline()
	dbprintdisplay.addhistdisplay(10,'dbprintbuffer')

	return infodisplay,statusdisplay,logdisplay,dbprintdisplay

class Dashboard0(Dashboard):

	def __init__(self):
		clear()
		super().__init__()
		
		self.width=os.get_terminal_size()[0]-1
		infodisplay,logdisplay,dbprintdisplay=get3displays(self.width)

		self.add_display(infodisplay,0,0)
		self.add_display(logdisplay,25,0)
		self.add_display(dbprintdisplay,25,self.width//2)

		self.draw_all()

	





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



