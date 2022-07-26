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
import jax.random as rnd
import sys
import util
import copy
from collections import deque
import sys


lossfn=util.sqloss
heavy_threshold=8
BOX='\u2588'
box='\u2592'
bar='\u2015'



def now(timesep=':'):
	date,time=str(datetime.datetime.now()).split('.')[0].split(' ')
	return date,time.replace(':',timesep)


def nowstr():
	date,time=now(timesep=' ')
	return date+' time '+time


session_ID=nowstr()


#====================================================================================================
# tracking
#====================================================================================================

class Tracker:

	def __init__(self):
		self.autosavepaths=set()
		self.t0=time.perf_counter()

		self.trackedvals=dict()
		self.listeners=[]

		self.ID=nowstr()

	def add_listener(self,listener):
		self.listeners.append(listener)
		
	def add_autosavepaths(self,*paths):
		self.autosavepaths.update(paths)

	def set(self,name,val):
		self.trackedvals[name]=(self.timestamp(),val)
		self.poke(name,val)

	def get(self,name):
		return self.trackedvals[name][1]

	def poke(self,name,val):
		for l in self.listeners:
			l.poke(name,val)

	"""
	# only register once
	"""
	def register(self,obj,varnames):
		for name in varnames:
			self.set(name,vars(obj)[name])

	def timestamp(self):
		return time.perf_counter()-self.t0

	def save(self,path):
		save(path,trackedvals)
		
	def autosave(self):
		for path in self.autosavepaths:
			self.save(path)
		



class HistTracker(Tracker):
	
	def __init__(self):
		Tracker.__init__(self)
		self.hists=dict()
		self.tracked_objs=dict()

	def set(self,name,val):
		if name not in self.hists:
			self.hists[name]={'timestamps':[],'vals':[]}
		self.hists[name]['timestamps'].append(self.timestamp())
		self.hists[name]['vals'].append(val)
		self.trackedvals[name]=(self.timestamp(),val)
		self.poke(name,val)

	def log(self,msg):
		msg='{} | {}'.format(nowstr(),msg)
		self.set('log',msg)
		log(msg,ID=self.ID)

	def gethist(self,name,timestamps=False):
		return self.hists[name] if timestamps else self.hists[name]['vals']

	def gethists(self):
		return {name:self.gethist(name) for name in self.hists}

	def save(self,path):
		save(path,self.hists)







def getvarhist(path,varname):
	varhist=get(path)[varname]
	return varhist['timestamps'],varhist['vals']

def getlastval(path,varname):
	return get(path)[varname]['vals'][-1]






#====================================================================================================


def log(msg,ID=session_ID):
	savetxt('logs/'+ID,msg+'\n')


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


#====================================================================================================

keys=[rnd.PRNGKey(0)]
keys=deque(keys)

def nextkey():
	keys=globals()['keys']
	if len(keys)==1:
		_,*keys=rnd.split(keys[0],1000)
		globals()['keys']=deque(keys)
	return globals()['keys'].popleft()




#====================================================================================================

def mkdir(path):
	try:
		os.mkdir(path)
	except OSError:
		pass


def makedirs(filepath):
	path='/'.join(filepath.split('/')[:-1])
	filename=filepath.split('/')[-1]
	os.makedirs(path,exist_ok=True)	

def save(path,data):
	makedirs(path)
	with open(path,'wb') as file:
		pickle.dump(data,file)

def savefig(path,fig=None):
	makedirs(path)
	if fig==None:
		plt.savefig(path)
	else:
		fig.savefig(path)

def savetxt(path,msg,mode='a'):
	makedirs(path)
	with open(path,mode) as f:
		f.write(msg)
	
def get(path):
	with open(path,"rb") as file:
		data=pickle.load(file)
	return data



#====================================================================================================

def formatvars(elements,separator=' ',ignore={}):
	return separator.join(['{}={}'.format(name,val) for name,val in elements if name not in ignore])



def castval(val):
	for f in [int,cast_str_as_list_(int),float,cast_str_as_list_(float)]:
		try:
			return f(val)
		except:
			pass
	return val


def cast_str_as_list_(dtype):
	def cast(s):
		return [dtype(x) for x in s[1:-1].split(',')]
	return cast


def parsedef(s):
	name,val=s.split('=')
	return name,castval(val)
		

def get_cmdln_args():
	cmdargs=sys.argv[1:]
	defs=dict([parsedef(_) for _ in cmdargs])
	return defs



def orderedunion(A,B):
	A,B=list(A),deque(B)
	S=set(A)
	for b in B:
		if b not in S:
			A.append(b)
			S.add(b)
	return A
		





#====================================================================================================





# testing

if __name__=='__main__':
	for i in range(10):
		print(nextkey())


