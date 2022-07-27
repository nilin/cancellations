import sys
import os
import math
import pdb
import config as cfg
from config import getdefault_histtracker


#----------------------------------------------------------------------------------------------------


BOX='\u2588'
box='\u2592'
dash='\u2015'


#----------------------------------------------------------------------------------------------------
# display tracked values
#----------------------------------------------------------------------------------------------------




class AbstractSlate:

	def __init__(self,tracker=None):
		self.elements=[]
		self.ln=1
		self.tracker=getdefault_histtracker() if tracker==None else tracker
		self.tracker.add_listener(self)

	def add(self,display,height=1):
		self.elements.append((self.ln,display))
		self.ln=self.ln+height

	def poke(self,*args):
		if 'log' in args or 'refresh' in args:
			self.refresh()

	def refresh(self):
		self.draw()

	def addbar(self,fn,**kwargs):
		self.add(Bar(fn,self,**kwargs))

	def addtext(self,msg,height=1):
		self.add(Text(msg,self,height=height),height)

	def addspace(self,n=1):
		self.ln=self.ln+n

	@staticmethod
	def cols():
		return os.get_terminal_size()[0]-1

	
		
class DisplayElement:
	def __init__(self,fn,slate,tracker=None,height=1,**kwargs):
		self.fn=fn
		self.slate=slate
		self.tracker=slate.tracker if tracker==None else tracker
		self.kwargs=kwargs
		self.height=height

	def getstr_safe(self):
		try:
			s=self.getstr()
		except Exception as e:
			s='pending...{}'.format(str(e))
		return '\n'.join([l+(self.slate.cols()-len(l))*' ' for l in s.splitlines()[-self.height:]])

class Bar(DisplayElement):
	def getstr(self):
		val=self.fn(self.tracker)
		return barstring(val,self.slate.cols(),**self.kwargs)
	
class Text(DisplayElement):

	def getstr(self):
		if type(self.fn)==str:
			return self.fn
		else:
			msg=self.fn(self.tracker)
		return '\n'.join(msg) if type(msg)==list else msg






def barstring(val,fullwidth,style=BOX,emptystyle=' ',**kwargs):

	barwidth=math.floor(fullwidth*min(val,1))
	remainderwidth=fullwidth-barwidth

	return (barwidth-1)*style+BOX+remainderwidth*emptystyle








#====================================================================================================




class Slate(AbstractSlate):

	def __init__(self,*args,**kwargs):
		super().__init__()
		self.clear()

	@staticmethod
	def clear():
		os.system('cls' if os.name == 'nt' else 'clear')

	@staticmethod
	def gotoline(n):
		print('\x1b['+str(n)+';0H')

	def draw(self):
		for ln,element in self.elements:
			self.gotoline(ln)
			print(element.getstr_safe())






"""
# display on a single line (when none of the packages work)
"""

class MinimalSlate(AbstractSlate):

	def __init__(self,*args,**kwargs):
		super().__init__()
		print(5*'\n')

	def cols(self):
		return self.elementwidth()

	def elementwidth(self):
		return os.get_terminal_size()[0]//len(self.elements)-3

	def draw(self):
		print((' {} '.format(cfg.box)).join([element.getstr_safe().replace('\n','|')[:self.elementwidth()] for _,element in self.elements]),end='\r')




#====================================================================================================


if __name__=='__main__':
	import time

	print('\ntest minimal slate (single line)')
	print('display on a single line (when none of the packages work)\n')


	s=MinimalSlate()
	tk=s.tracker
	
	s.addtext('x')
	s.addbar(lambda tk:tk.get('x')/100)
	s.addtext('y')
	s.addbar(lambda tk:1-tk.get('y')/100)


	for Y in range(100):
		tk.set('y',Y)
		for X in range(100):
			tk.set('x',X)
			time.sleep(.001)
	






