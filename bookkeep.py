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
import copy
from collections import deque




def now(timesep=':'):
	date,time=str(datetime.datetime.now()).split('.')[0].split(' ')
	return date,time.replace(':',timesep)


def nowstr():
	date,time=now(timesep=' ')
	return date+' time '+time




#====================================================================================================
# tracking
#====================================================================================================

class Tracker:

	def __init__(self):
		self.autosavepaths=[]
		self.t0=time.perf_counter()

		self.trackedvals=dict()
		self.listeners=[]

		self.ID=nowstr()

	def add_listener(self,listener):
		self.listeners.append(listener)
		
	def add_autosavepath(self,path):
		self.autosavepaths.append(path)

	def set(self,name,val):
		self.trackedvals[name]=(self.timestamp(),val)
		self.refresh(name,val)

	def get(self,name):
		return self.trackedvals[name][1]

	def getvals(self):
		return {name:self.get(name) for name in self.trackedvals}

	def refresh(self,name,val):
		for l in self.listeners:
			l.refresh(name,val)

	"""
	# only register once
	"""
	def register(self,obj,varnames):
		for name in varnames:
			self.set(name,vars(obj)[name])

	def timestamp(self):
		return time.perf_counter()-self.t0

	def save(self,path):
		save(trackedvals,path)
		
	def autosave(self):
		for path in self.autosavepaths:
			self.save(path)
		



class HistTracker(Tracker):
	
	def __init__(self):
		Tracker.__init__(self)
		self.hists=dict()
		self.schedule=standardschedule
		self.tracked_objs=dict()

	def set(self,name,val):
		if name not in self.hists:
			self.hists[name]={'timestamps':[],'vals':[]}
		self.hists[name]['timestamps'].append(self.timestamp())
		self.hists[name]['vals'].append(val)
		self.trackedvals[name]=(self.timestamp(),val)
		self.refresh(name,val)

	def gethist(self,name):
		return self.hists[name]['vals']

	def gethists(self):
		return {name:self.gethist(name) for name in self.hists}

	def save(self,path):
		save(self.hists,path)







def getvarhist(path,varname):
	varhist=get(path)[varname]
	return varhist['timestamps'],varhist['vals']

def getlastval(path,varname):
	return get(path)[varname]['vals'][-1]






class EmptyDashboard:

	def __init__(self):
		self.n=0

	def refresh(self,defs):
		print('empty dashboard call number {}'.format(self.n),end='\r')
		self.n=self.n+1
		pass


emptydashboard=EmptyDashboard()
bgtracker=Tracker()


#====================================================================================================
# 
#====================================================================================================


def arange(*args):
	return list(range(*args))

standardschedule=deque(arange(60)+arange(60,600,5)+arange(600,3600,60)+arange(3600,3600*24,300))


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
		elapsed=self.elapsed()
		self.time=time.perf_counter()
		return elapsed

	def elapsed(self):
		return time.perf_counter()-self.time


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



