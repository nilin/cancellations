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
import datetime
import sys


lossfn=util.sqloss
heavy_threshold=8
BOX='\u2588'
box='\u2592'
dash='\u2015'



def now():
	return str(datetime.datetime.now()).split('.')[0].split(' ')


def nowstr():
	date,time=now()
	date='-'.join(date.split('-')[1:])
	time=''.join([x for pair in zip(time.split(':'),['h','m','s']) for x in pair])
	return date+'|'+time

def timestamp():
	return time.perf_counter()


#====================================================================================================
# tracking
#====================================================================================================

t0=time.perf_counter()
trackedvals=dict()
hists=dict()
eventlisteners=dict()
sessionID=nowstr()


outpaths=set()

def histpaths():
	return [path+'hist' for path in outpaths]

def logpaths():
	return [path+'log' for path in outpaths]+['logs/'+sessionID]


#----------------------------------------------------------------------------------------------------

def addlistener(listener,*signals):
	for signal in signals:
		if signal not in eventlisteners: eventlisteners[signal]=set()
		eventlisteners[signal].add(listener)
		
def trackcurrent(name,val):
	trackedvals[name]=(timestamp(),val)
	pokelisteners(name)

def setstatic(name,val):
	assert name not in trackedvals
	trackcurrent(name,val)

def getval(name):
	return trackedvals[name][1]

def pokelisteners(signal,*args):
	if signal in eventlisteners:	
		for listener in eventlisteners[signal]:
			listener.poke(signal,*args)

#----------------------------------------------------------------------------------------------------


def trackhist(name,val):
	if name not in hists:
		hists[name]={'timestamps':[],'vals':[]}
	hists[name]['timestamps'].append(timestamp())
	hists[name]['vals'].append(val)
	trackcurrent(name,val)

def savehist(*paths):
	save(hists,*paths)
		
def autosave():
	savehist(*histpaths())

def gethist(name,timestamps=False):
	return hists[name] if timestamps else hists[name]['vals']

def gethists():
	return hists


#----------------------------------------------------------------------------------------------------

def log(msg):
	msg='{} | {}'.format(datetime.timedelta(seconds=int(timestamp())),msg)
	trackhist('log',msg)
	write(msg+'\n',*logpaths())
	pokelisteners('log')


def debuglog(msg):
	write(str(msg)+'\n\n\n','debug/'+sessionID)

#----------------------------------------------------------------------------------------------------
#====================================================================================================


def retrievevarhist(path,varname):
	varhist=get(path)[varname]
	return varhist['timestamps'],varhist['vals']

def retrievelastval(path,varname):
	return get(path)[varname]['vals'][-1]



#====================================================================================================



class Stopwatch:
	def __init__(self):
		self.time=0
		self.tick()

	def tick(self):
		t=self.elapsed()
		self.time=time.perf_counter()
		return t

	def elapsed(self):
		return time.perf_counter()-self.time

	def reset_after(self,timebound):
		if self.elapsed()>timebound:
			self.tick()
			return True
		else:
			return False


arange=lambda *ab:list(range(*ab))

defaultsched=jnp.array(arange(5)+arange(5,20,5)+arange(20,60,10)+arange(60,300,30)+arange(300,600,60)+arange(600,3600,300)+arange(3600,24*3600,3600))


class Scheduler:
	def __init__(self,sched=defaultsched):
		self.sched=deque(copy.deepcopy(sched))
		self.t0=time.perf_counter()

	def elapsed(self):
		return time.perf_counter()-self.t0

	def dispatch(self):
		disp=False
		t=self.elapsed()
		while t>self.sched[0]:
			self.sched.popleft()
			disp=True
		return disp
		
		


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

def makedirs(filepath):
	path='/'.join(filepath.split('/')[:-1])
	filename=filepath.split('/')[-1]
	os.makedirs(path,exist_ok=True)	

def save(data,*paths):
	for path in paths:
		makedirs(path)
		with open(path,'wb') as file:
			pickle.dump(data,file)
	log('Saved data to {}'.format(paths))

def savefig(*paths,fig=None):
	for path in paths:
		makedirs(path)
		if fig==None:
			plt.savefig(path)
		else:
			fig.savefig(path)
	log('Saved figure to {}'.format(paths))

def write(msg,*paths,mode='a'):
	for path in paths:
		makedirs(path)
		with open(path,mode) as f:
			f.write(msg)
	
def retrieve(path):
	with open(path,"rb") as file:
		return pickle.load(file)

		


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
		return [dtype(x) for x in s.split(',')]
	return cast


def parsedef(s):
	name,val=s.split('=')
	return name,castval(val)
		

def parse_cmdln_args(cmdargs=sys.argv[1:]):
	cmdargs=deque(cmdargs)
	args=[]
	while len(cmdargs)>0 and '=' not in cmdargs[0]:
		args.append(cmdargs.popleft())

	defs=dict([parsedef(_) for _ in cmdargs])
	return args,defs



def orderedunion(A,B):
	A,B=list(A),deque(B)
	S=set(A)
	for b in B:
		if b not in S:
			A.append(b)
			S.add(b)
	return A
		

def terse(l):
	return str([round(float(e*1000))/1000 for e in l])

#====================================================================================================





# testing

if __name__=='__main__':
	for i in range(10):
		print(nextkey())


