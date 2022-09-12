import os
import math
from ..utilities import config as cfg, tracking,textutil
from ..utilities.tracking import session
import collections
from ..utilities.textutil import BOX,box,dash,infty

#----------------------------------------------------------------------------------------------------



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
		return text
#		Lines=text.splitlines()
#		if hasattr(self,'width'):
#			lines=[]
#			for Line in Lines:
#
#				indent=len(Line)
#				Line=Line.lstrip()
#				indent-=len(Line)
#				indent=indent*' '
#
#				while True:
#					lines.append(indent+Line[:self.width])
#					Line=Line[self.width:]
#					if Line=='':break
#			return '\n'.join(lines)
#		return text


	def _gettext_(self):
		raise NotImplementedError

	def setwidth(self,width):
		self.width=width

	


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
		return tracking.currentprocess().getval(self.query)

class LogDisplay(Display):
	def __init__(self,**kw):
		super().__init__(bottom=True,**kw)
		#super().__init__(wrap=True,**kw)
	def _gettext_(self):
		rlog=session.gethist('recentlog')
		return '\n'.join(rlog)


#----------------------------------------------------------------------------------------------------


class CompositeDisplay(Display):
	def __init__(self,xlim,ylim,*a,**kw):
		x0,x1=xlim
		y0,y1=ylim
		
		super().__init__(*a, \
						 xlim=xlim, ylim=ylim,
						 width=x1-x0, height=y1-y0,
						 elements=tracking.dotdict(), \
						 defaultnames=collections.deque(range(100)), **kw)

	def __getattr__(self,name):
		try: return super.__getattr__(name)
		except: return self.elements.__getattr__(name)

	def add(self,e,name=None):
		if name==None:name=self.defaultnames.popleft()
		self.elements[name]=e
		return e,name

	def element(self,e):
		if e in self.elements.keys(): return self.elements[e]
		else: return e

	def delkeys(self,*keys):
		for k in keys:
			del self.elements[k]




class StackedDisplay(CompositeDisplay):
	def _gettext_(self):
		return '\n'.join([e.gettext() for e in self.elements.values()])

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
		return tracking.currentprocess().getcurrentval(self.query)


class NumberDisplay(QueryDisplay):
	def __init__(self,query,**kw):
		super().__init__(query=query,**kw)

		if 'avg_of' in kw:
			self.hist=tracking.History()
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
	def __init__(self,query,style=BOX,emptystyle=dash,**kw):
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



#----------------------------------------------------------------------------------------------------

def wraptext(msg,style=dash):
    width=max([len(l) for l in msg.splitlines()])
    line=dash*width
    return '{}\n{}\n{}'.format(line,msg,line)

def wraplines(lines,style=dash):
    width=max([len(l) for l in lines])
    line=dash*width
    return [line]+lines+[line]
