import colorama
colorama.init()
import sys
import os
import math
import pdb
from bookkeep import HistTracker


#----------------------------------------------------------------------------------------------------

def gotoline(n):
	print('\x1b['+str(n)+';0H')

def clear():
	os.system('cls' if os.name == 'nt' else 'clear')


BOX='\u2588'
box='\u2592'



#----------------------------------------------------------------------------------------------------
# display tracked values
#----------------------------------------------------------------------------------------------------




class Dashboard:

	def __init__(self):
		self.elements=[]
		self.ln=1
		#self.defs=dict()
		self.tracker=HistTracker()
		clear()

	def add(self,*displays):
		for display in displays:
			self.elements.append((self.ln,display))
			self.ln=self.ln+1

	def refresh(self,name,val):
		#self.defs[name]=val
		self.tracker.set(name,val)
		for ln,element in self.elements:
			gotoline(ln)
			print(element.getprint(self.tracker.getvals(),self.tracker.gethists()))

	def addbar(self,fn):
		self.add(Bar(fn))

	def addtext(self,*msgs):
		for msg in msgs:
			self.add(Text(msg))

	def addspace(self,n=1):
		self.ln=self.ln+n

	
		
class Display:
	def getprint(self,defs,hists):
		try:
			s=self.tryprint(defs,hists)
		except Exception as e:
			s='pending...{}'.format(str(e))
		s=s+' '*(os.get_terminal_size()[0]-len(s))
		return s

class Bar(Display):
	def __init__(self,valfn):
		self.valfn=valfn

	def tryprint(self,defs,hists):
		val=self.valfn(defs,hists)
		return barstring(val)
	
class Text(Display):
	def __init__(self,msg):
		self.msg=msg

	def tryprint(self,defs,hists):
		msg=self.msg
		return msg if type(msg)==str else str(msg(defs,hists))

#- bars ------------------------------------------------------------------------------------------------------------------------

def barstring(val,fullwidth=None,style=BOX,emptystyle=' '):

	if fullwidth==None:
		fullwidth=os.get_terminal_size()[0]

	barwidth=math.floor((fullwidth-1)*min(val,1))
	remainderwidth=fullwidth-barwidth

	Style=''	
	EmptyStyle=''
	while len(Style)<barwidth:
		Style=Style+style
	while len(EmptyStyle)<barwidth:
		EmptyStyle=EmptyStyle+emptystyle

	return Style[:barwidth]+BOX+remainderwidth*emptystyle[:remainderwidth]







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
	






