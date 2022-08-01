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
import re




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



#----------------------------------------------------------------------------------------------------
# temporary values

def addlistener(listener,*signals):
	for signal in signals:
		if signal not in eventlisteners: eventlisteners[signal]=set()
		eventlisteners[signal].add(listener)
		
def trackcurrent(name,val):
	trackedvals[name]=(timestamp(),val)
	pokelisteners(name)

def getval(name):
	try:
		return trackedvals[name][1]
	except:
		return sessionstate.static[name]

def pokelisteners(signal,*args):
	if signal in eventlisteners:	
		for listener in eventlisteners[signal]:
			listener.poke(signal,*args)


#----------------------------------------------------------------------------------------------------


def histpaths():
	return [path+'hist' for path in outpaths]

def logpaths():
	return [path+'log' for path in outpaths]+['logs/'+sessionID]



class State:
	def __init__(self,static=None,hists=None):
		self.static=dict() if static==None else static
		self.hists=dict() if hists==None else hists

	def remember(self,name,val,t=None):
		if t==None:t=timestamp()
		self.initentry(name)
		self.hists[name]['timestamps'].append(t)
		self.hists[name]['vals'].append(val)

	def save(self,*paths):
		save({'static':self.static,'hists':self.hists},*paths)

	def gethist(self,name):
		return self.hists[name]['timestamps'],self.hists[name]['vals']

	def initentry(self,name):
		if name not in self.hists:
			self.hists[name]={'timestamps':[],'vals':[]}

	def linkentry(self,name):
		self.initentry(name)
		return self.hists[name]

	def getlinks(self,*names):
		for name in names:
			self.initentry(name)
		return {name:self.hists[name] for name in names}

	def clonefrom(self,path):
		D=load(path)
		self.static=D['static']
		self.hists=D['hists']

#
class LoadedState:
	def __init__(self,path):
		self.clonefrom(path)


def retrievestate(path):
	print('\nloading from \n'+path+'\n')
	data=load(path)
	globals()['sessionstate']=State(*[data[k] for k in ['static','hists']])


		
#
#
#def loadstate(path):
#	state,hists=[get(path)[k] for k in ['state','hists']]
#	return State(state=state,hists=hists)

	
def getrecentlog(n):
	return sessionstate.gethist('log')[1][-n:]

def get_errlog():
	try:
		return sessionstate.gethist('errlog')[1]
	except:
		return []

def setstatic(name,val):
#	try:
#		assert name not in sessionstate.static
#	except:
#		dblog(name)
	sessionstate.static[name]=val
	#log('{}={}'.format(name,val))

def register(lcls,*names):
	for name in names:
		setstatic(name,lcls[name])

def remember(name,val):
	sessionstate.remember(name,val)
	trackcurrent(name,val)

def savestate(*paths):
	sessionstate.save(*paths)
		
def autosave():
	savestate(*histpaths())

def loadvarhist(path,varname):
	tempstate=loadstate(path)
	hist=tempstate.gethist(varname)
	del tempstate
	return hist




#----------------------------------------------------------------------------------------------------

def log(msg):
	msg='{} | {}'.format(datetime.timedelta(seconds=int(timestamp())),msg)
	remember('log',msg)
	write(msg.replace('\n','|')+'\n',*logpaths())
	if trackduration:
		write(str(int(timestamp())),*[os.sep.join(pathlog.split(os.sep)[:-1])+os.sep+'duration' for pathlog in logpaths()],mode='w')	
	pokelisteners('log')

def dblog(msg):
	dbprint(msg)
	write(str(msg)+'\n','debug/'+sessionID)

def errlog(msg):
	write(str(msg)+'\n\n\n','debug/errordump '+sessionID)

def dbprint(msg):
	dbprintbuffer.append(msg)

#----------------------------------------------------------------------------------------------------
#====================================================================================================


#====================================================================================================


class Timeup(Exception): pass

arange=lambda *ab:list(jnp.arange(*ab))

#def defaultsched(timebound):
#	jnp.array([5]+arange(0,60,10)+arange(60,300,30)+arange(300,600,60)+arange(600,hour,300)+arange(hour,timebound,hour))

def expsched(step1,timebound,percentincrease=.1,skipzero=False):
	delta=jnp.log(1+percentincrease)
	t1=step1/delta
	return jnp.concatenate([jnp.arange(step1 if skipzero else 0,t1,step1),jnp.exp(jnp.arange(jnp.log(t1),jnp.log(timebound),delta)),jnp.array([timebound])])

def periodicsched(step,timebound,skipzero=False):
	return jnp.array(arange(step if skipzero else 0,timebound,step)+[timebound])

def stepwiseperiodicsched(stepsizes,transitions):
	return jnp.concatenate([jnp.arange(transitions[i],transitions[i+1],step) for i,step in enumerate(stepsizes)]+[jnp.array([transitions[-1]])])



class Scheduler:
	def __init__(self,schedule):
		self.schedule=schedule
		self.rem_sched=deque(copy.deepcopy(schedule))
		self.t0=time.perf_counter()

	def elapsed(self):
		return time.perf_counter()-self.t0

	def dispatch(self,t=None):
		if t==None:t=self.elapsed()
		disp=False
		while t>self.rem_sched[0]:
			disp=True
			self.rem_sched.popleft()
			if len(self.rem_sched)==0:
				raise Timeup
		return disp

		
	def filter(self,times,*valueshists):
		timeticks=[times[0]]
		filteredhists=[[hist[0]] for hist in valueshists]
		try:
			for t,*values in list(zip(times,*valueshists))[1:]:
				if self.dispatch(t):
					timeticks.append(t)
					for val,hist in zip(values,filteredhists):
						hist.append(val)
		except Timeup:
			pass
		return (timeticks,*filteredhists)
		#return (timeticks,)+filteredhists


def filterschedule(sched,times,*valueshist):
	sc=Scheduler(sched)
	out=sc.filter(times,*valueshist)	
	del sc
	return out


#====================================================================================================


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


def savefig_(pathsuffixes,fig=None):
	savefig(*[path+pathsuffixes for path in outpaths],fig=fig)

def write(msg,*paths,mode='a'):
	for path in paths:
		makedirs(path)
		with open(path,mode) as f:
			f.write(msg)
	
def load(path):
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

def selectone(options,l):
	choice=list(options.intersection(l))
	assert(len(choice)==1)
	return choice[0]


#====================================================================================================


def longestduration(folder):
	def relorder(subfolder):
		try:
			with open(folder+'/'+subfolder+'/duration','r') as f:
				return int(f.read())
		except:
			return -1
	return folder+max([(subfolder,relorder(subfolder)) for subfolder in os.listdir(folder)],key=lambda pair:pair[1])[0]
	
def latest(folder):
	folders=[f for f in os.listdir(folder) if len(f)==15 and len(re.sub('[^0-9]','',f))==10]
	def relorder(subfoldername):
		return int(re.sub('[^0-9]','',subfoldername))
	return folder+max([(subfolder,relorder(subfolder)) for subfolder in folders],key=lambda pair:pair[1])[0]


lossfn=util.sqloss
heavy_threshold=8
BOX='\u2588'
box='\u2592'
dash='\u2015'

t0=time.perf_counter()
trackedvals=dict()
eventlisteners=dict()
sessionID=nowstr()
outpaths=set()
trackduration=False

sessionstate=State()
dbprintbuffer=['']

keys=[rnd.PRNGKey(0)]
keys=deque(keys)

hour=3600
day=24*hour
week=7*day

cmdparams,cmdredefs=parse_cmdln_args()

# testing

if __name__=='__main__':

#	print(stepwiseperiodicsched([10,100],[0,60,600]))
#	for i in range(10):
#		print(nextkey())

	print(selectone({'r','t'},[1,4,'r',5,'d']))
