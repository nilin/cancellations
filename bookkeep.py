import numpy as np
import math
import pickle
import time
import copy
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import datetime
import pdb
import sys

import screendraw

#from util import str_


def str_(*args):
	return ''.join([str(x) for x in args])
def log(*args,**kwargs):

	msg=str_(*args)

	#msg=msg+'\n\n-'+time.ctime(time.time())+'\n\n'
	with open('log','a') as f:
		f.write('\n'+str(msg))
	print(msg,**kwargs)


class Stopwatch:
	def __init__(self):
		self.time=0
		self.tick()

	def tick(self):
		elapsed=time.perf_counter()-self.time
		self.time=time.perf_counter()
		return elapsed


def mkdir(path):
	try:
		os.mkdir(path)
	except OSError:
		pass


def savefig(path,fig=None):
	makedirs(path)
	if fig==None:
		plt.savefig(path)
	else:
		fig.savefig(path)

def save(data,path):
	makedirs(path)
	with open(path,'wb') as file:
		pickle.dump(data,file)


def now(timesep=':'):
	date,time=str(datetime.datetime.now()).split('.')[0].split(' ')
	return date,time.replace(':',timesep)


def nowstr():
	date,time=now(timesep=' ')
	return date+' time '+time

def nowpath(toplevelfolder,fn=''):
	tl=toplevelfolder
	return (tl if tl[-1]=='/' else tl+'/')+nowstr()+'/'+fn


def makedirs(filepath):
	path='/'.join(filepath.split('/')[:-1])
	filename=filepath.split('/')[-1]
	os.makedirs(path,exist_ok=True)	
	
def get(path):
	with open(path,"rb") as file:
		data=pickle.load(file)
	return data
	

def getdata(filename):
	return get('data/'+filename)


			
BOX='\u2588'
box='\u2592'
bar='\u2015'


def printemph(s):
	s=10*box+' '+s+' '+200*box
	print(s[:150])
	






def formatvars(elements,separator=' ',ignore={}):
	return separator.join([s+'='+str(v) for s,v in elements.items() if s not in ignore])




def castval(val):
	try:
		return int(val)
	except:
		pass
	try:
		return cast_str_as_list(val)
	except:
		return val


def cast_str_as_list(s):
	return [int(x) for x in s.split(',')]


def parsedef(s):
	try:
		name,val=s.split('=')
		return name,castval(val)
	except:
		return None,None
		

def getparams(globalvars,sysargv,requiredvars={}):
	cmdargs=sysargv[1:]
	defs=dict([parsedef(_) for _ in cmdargs])

	for name in requiredvars:
		if name not in defs and name not in globalvars:
			defs[name]=castval(input(name+'='))
	globalvars.update(defs)



#----------------------------------------------------------------------------------------------------
# track values
#----------------------------------------------------------------------------------------------------

trackedvals=dict()

	

def track(name,val):
	trackedvals[name]=val
	printdisplays()


#----------------------------------------------------------------------------------------------------
# display tracked values
#----------------------------------------------------------------------------------------------------




displays=[]


class Display:
	def print(self):
		try:
			self.tryprint()
		except Exception as e:
			printbar(1,style='...pending...'+str(e))

class Bar(Display):
	def __init__(self,valfn):
		self.valfn=valfn

	def tryprint(self):
		val=self.valfn(trackedvals)
		printbar(val)
	

class Text(Display):
	def __init__(self,msg):
		self._msg_=msg

	def tryprint(self):
		_msg_=self._msg_
		msg=_msg_ if type(_msg_)==str else _msg_(trackedvals)
		clearln()
		print(msg)

class Vspace(Display):
	def __init__(self,n):
		self.n=n

	def tryprint(self):
		for i in range(self.n):
			clearln()
			print()


def add(display):
	displays.append(display)

def addbar(valfn):
	add(Bar(valfn))

def addtext(msg):
	add(Text(msg))

def addspace(n):
	add(Vspace(n))

def printdisplays():

	screendraw.gotoline(0)
	for display in displays:
		display.print()




#- bars ------------------------------------------------------------------------------------------------------------------------

def barstring(val,outerwidth,style=BOX,emptystyle=' '):

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


# str, float
def printbar(val,**kwargs): 
	print(barstring(val,os.get_terminal_size()[0]-10,**kwargs))

def clearln():
	print((os.get_terminal_size()[0]-5)*' ',end='\r')
