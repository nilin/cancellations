import colorama
colorama.init()
import sys
import os
import math
import pdb
from config import getdefault_histtracker


#----------------------------------------------------------------------------------------------------

def gotoline(n):
	print('\x1b['+str(n)+';0H')

def clear():
	os.system('cls' if os.name == 'nt' else 'clear')


BOX='\u2588'
box='\u2592'
dash='\u2015'


#----------------------------------------------------------------------------------------------------
# display tracked values
#----------------------------------------------------------------------------------------------------




class Dashboard:

	def __init__(self,tracker=None):
		self.elements=[]
		self.ln=1
		self.tracker=getdefault_histtracker() if tracker==None else tracker
		self.tracker.add_listener(self)
		clear()

	def add(self,display):
		self.elements.append((self.ln,display))
		self.ln=self.ln+1

	def poke(self,*args):
		self.refresh()

	def refresh(self):
		for ln,element in self.elements:
			gotoline(ln)
			print(element.getprint())


	def addbar(self,fn,tracker=None,**kwargs):
		self.add(Bar(fn,self.tracker if tracker==None else tracker,**kwargs))

	def addtext(self,*msgs,tracker=None):
		for msg in msgs:
			self.add(Text(msg,self.tracker if tracker==None else tracker))

	def addlog(self,lines,tracker=None):
		self.addtext('log'+200*'.')
		self.add(Log(lines,self.tracker if tracker==None else tracker))
		self.addspace(lines+5)

	def addspace(self,n=1):
		self.ln=self.ln+n

	
		
class Display:
	def __init__(self,fn,tracker,**kwargs):
		self.fn=fn
		self.tracker=tracker
		self.kwargs=kwargs

	def getprint(self):
		try:
			s=self.tryprint()
		except Exception as e:
			s='pending...{}'.format(str(e))
		s=s+' '*(os.get_terminal_size()[0]-len(s))
		return s

class Bar(Display):
	def tryprint(self):
		val=self.fn(self.tracker)
		return barstring(val,**self.kwargs)
	
class Text(Display):
	def tryprint(self):
		msg=self.fn
		return msg if type(msg)==str else str(msg(self.tracker))

class Log(Display):
	def tryprint(self):
		lines=self.fn
		return (50*' '+'\n').join(self.tracker.gethist('log')[-lines:])


#- bars ------------------------------------------------------------------------------------------------------------------------

def barstring(val,fullwidth=None,style=BOX,emptystyle=' '):

	if fullwidth==None:
		fullwidth=os.get_terminal_size()[0]

	barwidth=math.floor((fullwidth-1)*min(val,1))
	remainderwidth=fullwidth-barwidth

	return barwidth*style+BOX+remainderwidth*emptystyle







#- test ------------------------------------------------------------------------------------------------------------------------


if __name__=='__main__':

	import time


	txt1=Text('x')
	bar1=Bar(lambda defs:defs['x'])
	txt2=Text('1-x')
	bar2=Bar(lambda defs:1-defs['x'])
	txt3=Text('y')
	bar3=Bar(lambda defs:1-defs['y'])

	dash=Dashboard()
	dash.add(txt1,bar1,txt2,bar2,txt3,bar3)

	for Y in range(100):
		for X in range(100):
			dash.refresh({'x':X/100,'y':Y/100})
			time.sleep(.001)
	






