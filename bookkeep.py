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
from collections import deque








#====================================================================================================
# tracking
#====================================================================================================

class Tracker:

	def __init__(self,**kwargs):
		self.autosavepaths=[]
		self.t0=time.perf_counter()

		self.passivelytracked=set()
		self.trackedvals=dict()
		self.dashboard=emptydashboard

		self.ID=nowstr()

	def latestvals(self):
		return {name:t_val[1] for name,t_val in self.trackedvals.items() if name!='timestamp'}

	def set_display_dash(self,dashboard):
		self.dashboard=dashboard
		
	def add_autosavepath(self,path):
		self.autosavepaths.append(path)

	def track(self,*names):
		self.passivelytracked.update(set(names))

	def set(self,name,val):
		self.trackedvals[name]=(self.timestamp(),val)
		self.refresh()

	def refresh(self):
		self.dashboard.refresh(self.latestvals())

	def register(self,*varnames):
		for name in varnames:
			self.set(name,vars(self)[name])

	def timestamp(self):
		return time.perf_counter()-self.t0

	def getpassivevals(self):
		return {name:(self.timestamp(),vars(self)[name]) for name in self.passivelytracked}

	def updateandgetvals(self):
		self.trackedvals.update(self.getpassivevals())
		self.trackedvals['timestamp']=self.timestamp()
		return self.trackedvals

	def save(self,path):
		save(self.updateandgetvals(),path)
		
	def autosave(self):
		for path in self.autosavepaths:
			self.save(path)
		



class HistTracker(Tracker):
	
	def __init__(self,**kwargs):
		super().__init__(**kwargs)
		self.hist=[]
		self.schedule=standardschedule

	def checkpoint(self):
		self.hist.append(self.updateandgetvals())
		self.autosave()

	def add_event(self,msg):
		self.set('event',msg)
		self.checkpoint()
	
	def save(self,path):
		hist=self.hist		

		save(hist,path)

	def poke(self):
		if self.time_elapsed<self.schedule[0]:
			self.schedule.popleft()
			self.checkpoint()

	def setvals(self,vals):
		super().setvals(vals)
		self.poke()






def getvarhist(path,varname):

	hist=get(path)
	timestamps,vals=zip(*[image[varname] for image in hist if varname in image])
	return timestamps,vals

def getlastval(path,varname):
	timestamp,val=get(path)[-1][varname]
	return val



class EmptyDashboard:
	def refresh(self,defs):
		print('empty dashboard')
		pass

emptydashboard=EmptyDashboard()




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


