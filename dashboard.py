import colorama
colorama.init()
import sys
import os
import math
import pdb


#----------------------------------------------------------------------------------------------------

def gotoline(n):
	print('\x1b['+str(n)+';0H')

def clear():
	os.system('cls' if os.name == 'nt' else 'clear')

#def clearln():
#	print((os.get_terminal_size()[0]-5)*' ',end='\r')


BOX='\u2588'
box='\u2592'

def maxwidth():
	return os.get_terminal_size()[0]-5



#----------------------------------------------------------------------------------------------------
# display tracked values
#----------------------------------------------------------------------------------------------------





class Dashboard:

	def __init__(self):
		self.elements=[]
		clear()

	def add(self,*displays):
		for display in displays:
			self.elements.append(display)

	def refresh(self,defs):
		gotoline(0)
		print(self.getdashprint(defs))

	def getdashprint(self,defs):
		return '\n'.join([element.getprint(defs) for element in self.elements])


	def addbar(self,fn):
		self.add(Bar(fn))

	def addtext(self,msg):
		self.add(Text(msg))

	def addspace(self,n=1):
		self.add(Vspace(n))

	
		
class Display:
	def getprint(self,defs):
		try:
			s=self.tryprint(defs)
		except Exception as e:
			s='pending...{}'.format(str(e))
		s=s+' '*(maxwidth()-len(s))
		return s

class Bar(Display):
	def __init__(self,valfn):
		self.valfn=valfn

	def tryprint(self,defs):
		val=self.valfn(defs)
		return barstring(val)
	
class Text(Display):
	def __init__(self,msg):
		self.msg=msg

	def tryprint(self,defs):
		msg=self.msg
		return msg if type(msg)==str else msg(defs)

class Vspace(Display):
	def __init__(self,n):
		self.n=n-1

	def tryprint(self,defs):
		return '\n'*self.n

#- bars ------------------------------------------------------------------------------------------------------------------------

def barstring(val,outerwidth=maxwidth(),style=BOX,emptystyle=' '):

	fullwidth=outerwidth-2
	barwidth=math.floor((fullwidth-1)*min(val,1))
	remainderwidth=fullwidth-barwidth

	Style=''	
	EmptyStyle=''
	while len(Style)<barwidth:
		Style=Style+style
	while len(EmptyStyle)<barwidth:
		EmptyStyle=EmptyStyle+emptystyle

	return '['+Style[:barwidth]+BOX+remainderwidth*emptystyle[:remainderwidth]+']'







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
	






