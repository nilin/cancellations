import os
import math
from re import L
from ..utilities import config as cfg, tracking,textutil,arrayutil
from ..utilities.tracking import Stopwatch, session
import collections
from collections import deque
import jax
import jax.numpy as jnp
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

		#for debug
		#text=self.msgtransform(self._gettext_()) if 'msgtransform' in vars(self) else self._gettext_()
		#if self.wrap: text=self.getwrapped(text)
		#return self.getcropped(text)

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

#class Runtext(Display):
#	def _gettext_(self):
#		return tracking.currentprocess().getval(self.query)

class LogDisplay(Display):
	def __init__(self,**kw):
		super().__init__(bottom=True,**kw)
		#super().__init__(wrap=True,**kw)
	def _gettext_(self):
		rlog=session.gethist('recentlog')
		return '\n'.join(rlog)


#----------------------------------------------------------------------------------------------------


class AbstractCompositeDisplay(Display):
	def __init__(self,*a,**kw):
		super().__init__(*a, elements=tracking.dotdict(), defaultnames=collections.deque(range(100)), **kw)

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
			if isinstance(self.elements[k],CompositeDisplay):
				self.elements[k].remove() 
			del self.elements[k]

	def remove(self):
		self.delkeys(*self.elements.keys())


class CompositeDisplay(AbstractCompositeDisplay):
	def __init__(self,xlim,ylim,*a,**kw):
		x0,x1=xlim
		y0,y1=ylim
		
		super().__init__(*a, xlim=xlim, ylim=ylim, width=x1-x0, height=y1-y0, **kw)



class StackedDisplay(CompositeDisplay):
	def _gettext_(self):
		return '\n'.join([e.gettext() for e in self.elements.values()])

	def add(self,e):
		if hasattr(self,'width'): e.setwidth(self.width)
		return super().add(e)

	def setwidth(self,width):
		self.width=width
		for e in self.elements: e.setwidth(width)



class SwitchDisplay(AbstractCompositeDisplay):
	def pickdisplay(self,name):
		self.activedisplay=name

	def _gettext_(self):
		return self.elements[self.activedisplay]._gettext_()

#----------------------------------------------------------------------------------------------------

def R_to_I_formatter(center,dynamicwidth):

	def parse_continuous(x):
		t=(x-center)/dynamicwidth
		return arrayutil.slowsigmoid_01(t)

	def parse(x,displaywidth):
		return math.floor(parse_continuous(x)*displaywidth)

	return parse

class Ticks(StaticText):
	def __init__(self,transform,ticks,labels=None,lstyle=' ',**kw):
		if labels==None: labels='|'*len(ticks)
		ticks,labels=zip(*sorted(zip(ticks,labels)))
		super().__init__(ticks=deque(ticks),\
			labels=deque(labels),transform=transform,lstyle=lstyle,**kw)

	def setwidth(self, width):
		super().setwidth(width)
		self.msg=''
		for i in range(width):
			if len(self.ticks)>0 and len(self.msg)>=self.transform(self.ticks[0],displaywidth=width):
				self.msg+=str(self.labels.popleft())
				self.ticks.popleft()
			else: self.msg+=self.lstyle


#----------------------------------------------------------------------------------------------------

class FlexDisplay(Display):
	def __init__(self,*queries,smoothing=None,parse):
		if smoothing==None: smoothing=[None for q in queries]
		super().__init__(queries=queries,smoothing=smoothing,parse=parse)
		self.reset()

	def reset(self):
		self.smoothers=[tracking.RunningAvgOrIden(s) for q,s in zip(self.queries,self.smoothing)]

	def getvals(self):
		return [s.update(tracking.currentprocess().getcurrentval(q)) for q,s in zip(self.queries,self.smoothers)]

	def _gettext_(self):
		return self.parse(self,*self.getvals())




class DynamicRange(Display,tracking.Stopwatch):
	def __init__(self,queryfn,customticks=None,customlabels=None):
		Display.__init__(self,queryfn=queryfn,customticks=customticks,customlabels=customlabels)
		Stopwatch.__init__(self)
		self.smoother=tracking.RunningAvg(100)

	def gettransform(self): 
		if not hasattr(self,'T') or self.tick_after(5):
			self.center=float(self.smoother.update(self.queryfn()))
			self.rangewidth=3*max([abs(self.smoother.avg()-t) for t in self.customticks])
			self.T=lambda t: R_to_I_formatter(self.center,self.rangewidth)(t,self.width)

	def _gettext_(self):
		self.gettransform()
		x=self.smoother.update(self.queryfn())

		moving=textutil.placelabels([self.T(x)],'|')

		ticks,prec=textutil.roundrangeandprecision(self.center,2*self.rangewidth,10)
		tickstr1=textutil.placelabels([self.T(t) for t in ticks],'|')
		labelstr1=textutil.placelabels([self.T(t) for t in ticks],['{:0.{}f}'.format(t,prec) for t in ticks])

		screenpos2=[self.T(t) for t in self.customticks]
		tickstr2=textutil.placelabels(screenpos2,'|')
		labelstr2=textutil.placelabels(screenpos2,self.customlabels)

		return moving.replace('|',textutil.BOX)+'\n'+\
			textutil.layer(self.width*textutil.dash,tickstr1,tickstr2,moving)+'\n'+\
			textutil.layer(tickstr2,labelstr1)+'\n'+\
			tickstr2+'\n'+\
			labelstr2

		#return textutil.overwrite(textutil.placelabels([T(t) for t in ticks],ticks),textutil.placelabels([T(self.getval())])




#----------------------------------------------------------------------------------------------------


class QueryDisplay(Display):
	def __init__(self,query,**kw):
		super().__init__(query=query,**kw)

	def getval(self):
		return tracking.currentprocess().getcurrentval(self.query)


class NumberDisplay(QueryDisplay):
	def __init__(self,query,avg_of=1,**kw):
		super().__init__(query=query,avg_of=avg_of,**kw)
		self.runningavg=tracking.RunningAvg(k=avg_of)

	def _gettext_(self):
		return self.formatnumber(self.runningavg.update(self.getval()))

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
