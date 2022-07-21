import numpy as np
import math
import pickle
import time
import copy
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import sys
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

def printbar(val,msg='',hold=True,printval=True,shift=0,**kwargs):

	msg=msg+(' '+'{:.4f}'.format(val) if printval else '')
	print(shift*' '+progressbar(val,140,**kwargs)+' '+msg,end=('\r' if hold else '\n'))



def printemph(s):
	s=10*box+' '+s+' '+200*box
	print(s[:150])
	


def progressbar(relwidth,fullwidth,style=BOX,emptystyle=bar):
	barwidth=int(math.floor(fullwidth*min(relwidth,1)))
	remainderwidth=fullwidth-barwidth-1

	Style=''	
	while len(Style)<barwidth:
		Style=Style+style

	return '['+Style[:barwidth]+BOX+remainderwidth*emptystyle+']'




def formatvars(elements):
	variables=dict()
	for e in elements:
		name,val=e.split('=')
		variables[name]=int(val)
	return variables


def formatvars_(elements):
	return ' '.join([s+'='+str(v) for s,v in elements.items()])



class Bars:
	def __init__(self):
		self.bars=dict()

	def setbar(self,name,val,txt):
		self.bars[name]=(val,txt)
		draw()

	def draw(self):
		data=[(val_txt[0],name+'='+str(val_txt[1])) for name,val_txt in self.bars.items()]
		printbars(data)



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

