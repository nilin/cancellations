from re import I
import sys
import os
import math
import numpy as np
import jax.numpy as jnp
import pdb
import config as cfg
from collections import deque
import time
import util
from config import session


#----------------------------------------------------------------------------------------------------


BOX='\u2588'
box='\u2592'
dash='\u2015'
infty='\u221E'

def clear():
	os.system('cls' if os.name == 'nt' else 'clear')

def print_at(y,x,msg):
	print('\x1b[{};{}H{}'.format(y,x,msg))


widthbound=250
line=dash*widthbound

#----------------------------------------------------------------------------------------------------



class Display:
	def __init__(self,**kw):
		for k,v in kw.items():
			setattr(self,k,v)

	def getlines(self):
		try:
			return self._getlines_()
		except Exception as e:
			return ['pending '+str(e)]

	def _getlines_(self):
		raise NotImplementedError


class StaticText(Display):
	def __init__(self,text):
		self.text=text
	def _getlines_(self):
		return self.text.splitlines()



class LogDisplay(Display):
	def __init__(self,height):
		self.height=height
	def _getlines_(self):
		lines=session.gethist('recentlog')[-(self.height-2):]
		return [line]+lines+[line]


#----------------------------------------------------------------------------------------------------

class StackedDisplay(Display):
	def __init__(self,**kw):
		super().__init__(**kw)
		self.displays=[]

	def _getlines_(self):
		return [line for display in self.displays for line in display.getlines()]

	def add(self,display):
		self.displays.append(display)
		if isinstance(display,DisplayElement):
			display.attach(self)
		return display

	def delete(self,*elements):
		for e in elements:
			self.displays.remove(e)

	def stack(self,*displays,style=4*'\n'):
		for display in displays:
			self.add(display)
			self.displays.append(StaticText(style))
		self.displays.pop()

	def addhistdisplay(self,query,**kw):
		self.add(HistDisplay(query,**kw))

	def addspace(self,n=1):	
		self.add(StaticText('\n'*n))


#----------------------------------------------------------------------------------------------------

class DisplayElement(Display):
	def __init__(self,query,**kw):
		super().__init__(query=query,**kw)

	def attach(self,container):
		self.container=container

	def getval(self): return self.container.memory.getcurrentval(self.query)
	def getwidth(self): return self.container.width

class ValDisplay(DisplayElement):
	def __init__(self,query,msg='{}',**kw):
		super().__init__(query,msg=msg,**kw)

	def _getlines_(self):
		return self.msg.format(self.getval()).splitlines()




class NumberDisplay(DisplayElement):
	def __init__(self,query,**kw):
		super().__init__(query=query,**kw)

		if 'avg_of' in kw:
			self.hist=cfg.History()
			self._getlines_=self.getlines1
		else:
			self._getlines_=self.getlines0

	def getlines0(self):
		out=self.getval()
		return self.formatnumber(out).splitlines()

	def getlines1(self):
		out=self.getval()
		self.hist.remember(out,membound=self.avg_of)
		histvals=self.hist.gethist()
		return self.formatnumber(sum(histvals)/len(histvals)).splitlines()


class Bar(NumberDisplay):
	def __init__(self,query,style=cfg.BOX,emptystyle=dash,**kw):

		Style=math.ceil(widthbound/len(style))*style
		Emptystyle=math.ceil(widthbound/len(emptystyle))*emptystyle

		super().__init__(query,Style=Style,Emptystyle=Emptystyle,**kw)

	def formatnumber(self,x):
		barwidth=math.ceil(self.getwidth()*max(min(x,1),0))
		return self.Style[:barwidth]+self.Emptystyle[barwidth:self.getwidth()+1]

class RplusBar(NumberDisplay):
	def formatnumber(self,x):
		s=[dash]*self.getwidth()
		s[0]='0'; s[self.getwidth()//2]='1'; s[-5:]='INFTY'
		t=math.floor(self.getwidth()*util.sigmoid(jnp.log(x)))
		s[t]=BOX
		return ''.join(s)

class NumberPrint(NumberDisplay):
	def __init__(self,query,msg='{:.3}',**kw):
		super().__init__(query,msg=msg,**kw)

	def formatnumber(self,x):
		return self.msg.format(x)



#----------------------------------------------------------------------------------------------------


class AbstractDashboard:

	def __init__(self):
		clear()
		self.displays=dict()
		self.defaultnames=deque(range(100))

	def add(self,display,xlim,ylim,name=None,draw=True):
		if name==None: name=self.defaultnames.popleft()
		cd=self.Concretedisplay(display,xlim,ylim)
		self.displays[name]=cd

		if draw: cd.draw()
		return cd

	def delete(self,name):
		del self.displays[name]

	def draw(self,name):
		self.displays[name].draw()

	def draw_all(self):
		for name,concretedisplay in self.displays.items():
			concretedisplay.draw()
