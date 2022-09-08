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
	def __init__(self,wrap=False,bottom=False,**kw):
		self.wrap=wrap
		self.bottom=bottom
		for k,v in kw.items():
			setattr(self,k,v)

	def getlines(self):
		return self.gettext().splitlines()

	def gettext(self):
		try:
			text=self.msgtransform(self._gettext_()) if 'msgtransform' in vars(self) else self._gettext_()
			if self.wrap: text=self.getwrapped(text)
			return self.getcropped(text)
		except Exception as e:
			return 'pending '+str(e)

	def getcropped(self,text):
		lines=text.splitlines()
		if hasattr(self,'width'):
			lines=[l[:self.width] for l in lines]
		if hasattr(self,'height'):
			if self.bottom: lines=lines[-self.height:]
			else: lines=lines[:self.height]
		return '\n'.join(lines)

	def getwrapped(self,text):
		Lines=text.splitlines()
		if hasattr(self,'width'):
			lines=[]
			for Line in Lines:

				indent=len(Line)
				Line=Line.lstrip()
				indent-=len(Line)
				indent=indent*' '

				while True:
					lines.append(indent+Line[:self.width])
					Line=Line[self.width:]
					if Line=='':break
			return '\n'.join(lines)
		return text


	def _gettext_(self):
		raise NotImplementedError

	def setwidth(self,width):
		self.width=width

	def attach(self,container):
		self.container=container
	


class StaticText(Display):
	def _gettext_(self):
		return self.msg

class VSpace(StaticText):
	def __init__(self, height, **kw):
		self.msg=height*'\n'
		super().__init__(**kw)

class Hline(StaticText):
	msg=line	

class SessionText(Display):
	def _gettext_(self):
		return session.getval(self.query)

class RunText(Display):
	def _gettext_(self):
		cprof=cfg.currentprofile()
		return cprof.run.getval(self.query)

class LogDisplay(Display):
	def __init__(self,**kw):
		super().__init__(bottom=True,**kw)
		#super().__init__(wrap=True,**kw)
	def _gettext_(self):
		rlog=session.gethist('recentlog')
		return '\n'.join(rlog)


#----------------------------------------------------------------------------------------------------


class CompositeDisplay(Display):
	def __init__(self,**kw):
		super().__init__(**kw)
		self.elements=[]
		self.names=dict()

	def add(self,e,name=None):
		self.elements.append(e)
		if name is not None: self.names[name]=e
		e.attach(self)
		return e

	def element(self,e):
		if e in self.names: return self.names[e]
		else: return e

	def delete(self,*elements):
		for e in elements:
			self.elements.remove(self.element(e))

	def drawelement(self,name):
		self.elements[self.element(name)].draw()

	def draw(self):
		for e in self.elements:
			e.draw()


class CDashboard(CompositeDisplay):

	def draw(self):
		cfg.screen.clear()
		cfg.screen.refresh()
		super().draw()


class StackedDisplay(CompositeDisplay):
	def _gettext_(self):
		return '\n'.join([e.gettext() for e in self.elements])

	def add(self,e):
		if hasattr(self,'width'): e.setwidth(self.width)
		return super().add(e)

	def setwidth(self,width):
		self.width=width
		for e in self.elements: e.setwidth(width)


#----------------------------------------------------------------------------------------------------


class QueryDisplay(Display):
	def __init__(self,query,**kw):
		super().__init__(query=query,**kw)

	def getval(self):
		cprof=cfg.currentprofile()
		return cprof.run.getcurrentval(self.query)


class NumberDisplay(QueryDisplay):
	def __init__(self,query,**kw):
		super().__init__(query=query,**kw)

		if 'avg_of' in kw:
			self.hist=cfg.History()
			self._gettext_=self._gettext_1
		else:
			self._gettext_=self._gettext_0

	def _gettext_0(self):
		out=self.getval()
		return self.formatnumber(out)

	def _gettext_1(self):
		out=self.getval()
		self.hist.remember(out,membound=self.avg_of)
		histvals=self.hist.gethist()
		return self.formatnumber(sum(histvals)/len(histvals))

	def formatnumber(self,x): raise NotImplementedError


class Bar(NumberDisplay):
	def __init__(self,query,style=cfg.BOX,emptystyle=dash,**kw):
		Style=math.ceil(widthbound/len(style))*style
		Emptystyle=math.ceil(widthbound/len(emptystyle))*emptystyle
		super().__init__(query,Style=Style,Emptystyle=Emptystyle,**kw)

	def formatnumber(self,x):
		barwidth=math.ceil(self.width*max(min(x,1),0))
		return self.Style[:barwidth]+self.Emptystyle[barwidth:self.width+1]

class RplusBar(NumberDisplay):
	def formatnumber(self,x):
		mapping=lambda x:1-1/(1+x)
		_mapping_=lambda x:round(math.floor(self.width*mapping(x)))
		s=_mapping_(x)*[BOX]+(self.width-_mapping_(x))*[dash]
		for i in [0,1,2,10]:
			s[_mapping_(i)]=str(i)
		return ''.join(s)

class NumberPrint(NumberDisplay):
	def __init__(self,query,msg='{:.3}',**kw):
		super().__init__(query,msg=msg,**kw)

	def formatnumber(self,x):
		return self.msg.format(x)



