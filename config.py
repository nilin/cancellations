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
import copy
from collections import deque
import datetime
import sys
import re














#----------------------------------------------------------------------------------------------------
# replacement for state
#----------------------------------------------------------------------------------------------------

class History:
	def __init__(self):
		self.snapshots=deque()

	def remember(self,val,metadata=None,norepeat=False,membound=None):
		if norepeat and self.getcurrentval()==val: pass

		if metadata==None: metadata=dict()
		self.snapshots.append((val,metadata|self.default_metadata()))
		if membound!=None and len(self.snapshots)>membound: self.snapshots.popleft()

	def gethist(self,*metaparams):
		valhist=[val for val,metadata in self.snapshots]
		metaparamshist=list(zip(*[[metadata[mp] for mp in metaparams] for val,metadata in self.snapshots]))
		return (valhist,*metaparamshist) if len(metaparams)>0 else valhist

	def getcurrentval(self):
		return self.snapshots[-1][0] if len(self.snapshots)>0 else None

	def default_metadata(self):
		return {'snapshot number':len(self.snapshots)}

	def filter(self,filterby,schedule):
		filteredhistory=History()
		schedule=deque(schedule)

		for val,metadata in self.snapshots:
			t=metadata[filterby]
			if t>=schedule[0]:
				filteredhistory.remember(val,metadata)
				while t>=schedule[0]:
					schedule.popleft()
					if len(schedule)==0: break
				if len(schedule)==0: break
		return filteredhistory


class BasicMemory:
	def __init__(self):
		self.hists=dict()

	def remember(self,name,val,metadata,**kwargs):
		if name not in self.hists:
			self.hists[name]=History()
		self.hists[name].remember(val,metadata,**kwargs)

	def trackcurrent(self,name,val,*x,**y):
		self.remember(name,val,*x,**y,membound=1)

	def log(self,msg):
		self.remember('log',msg)

	#def histref(self,name):
	#	return self.hists[name]	

	def gethist(self,name,*metaparams):
		return self.hists[name].gethist(*metaparams) #if name in self.hists else ([],)+tuple([[] for _ in metaparams])

	def getcurrentval(self,name):
		return self.hists[name].getcurrentval() #if name in self.hists else None

	def getval(self,name):
		return self.getcurrentval(name)


class Timer:
	def __init__(self,*x,**y):
		self.t0=time.perf_counter()

	def time(self):
		return time.perf_counter()-self.t0


class Memory(BasicMemory,Timer):
	def __init__(self):
		BasicMemory.__init__(self)
		Timer.__init__(self)
		self.memID=int(rnd.randint(nextkey(),minval=0,maxval=10**9,shape=(1,)))
		self.context=dict()
		self.listeners=[]

	def addcontext(self,name,val):
		self.context[name]=val
		self.trackcurrent(name,val)

	def getcontext(self):
		return self.context|{'memory {} time'.format(self.memID):self.time()}

	def remember(self,name,val,moremetadata=None,**kwargs):
		if moremetadata==None: moremetadata=dict()
		super().remember(name,val,self.getcontext()|moremetadata,**kwargs)
		self.pokelisteners(name)

	def addlistener(self,listener):
		self.listeners.append(listener)	

	def pokelisteners(self,*args):
		for l in self.listeners:
			l.poke(*args)


class ActiveMemory(Memory):

	def compute(self,queries,fn,outputname):
		inputvals=[self.getcurrentval(q) for q in queries]
		self.remember(outputname,fn(*inputvals))

	def computefromhist(self,queries,fn,outputname):
		inputvals=[self.gethist(q) for q in queries]
		self.remember(outputname,fn(*inputvals))





class Watched:
	def __init__(self):
		self.listeners=[]

	def addlistener(self,L):
		self.listeners.append(L)

	def pokelisteners(self,*args,**kwargs):
		for L in listeners:
			L.poke(*args,**kwargs)

#----------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------


def now():
	return str(datetime.datetime.now()).split('.')[0].split(' ')


def nowstr():
	date,time=now()
	date='-'.join(date.split('-')[1:])
	time=''.join([x for pair in zip(time.split(':'),['h','m','s']) for x in pair])
	return date+'|'+time



#
#----------------------------------------------------------------------------------------------------


def histpath():
	return outpath+'hist'

def logpaths():
	return [outpath+'log','logs/'+sessionID]



def register(lcls,*names):
	for name in names:
		setstatic(name,lcls[name])

#
#def savestate(*paths):
#	#sessionstate.save(*paths)
#		
#def save():
#	savestate(*histpaths())
#



#----------------------------------------------------------------------------------------------------

def log(msg):
	msg='{} | {}'.format(datetime.timedelta(seconds=int(session.time())),msg)
	session.log(msg)
	write(msg+'\n',*logpaths())
	if trackduration:
		write(str(int(session.time())),*[os.sep.join(pathlog.split(os.sep)[:-1])+os.sep+'duration' for pathlog in logpaths()],mode='w')	

def dblog(msg):
	write(str(msg)+'\n','debug/'+sessionID)

def dbprint(msg):
	session.remember('dbprintbuffer',str(msg),norepeat=True)


class Stopwatch:
	def __init__(self):
		self.t=time.perf_counter()-10**6

	def tick(self):
		t0=self.t
		self.t=time.perf_counter()
		return self.t-t0

	def dbtick(self,msg=''):
		dbprint('{} {}'.format(msg,self.tick()))



#----------------------------------------------------------------------------------------------------



arange=lambda *ab:list(jnp.arange(*ab))

#def expsched(step1,timebound,doublingtime=1,skipzero=False):
#	delta=jnp.log(2)/doublingtime
#	t1=step1/delta
#	return jnp.concatenate([jnp.arange(step1 if skipzero else 0,t1,step1),jnp.exp(jnp.arange(jnp.log(t1),jnp.log(timebound),delta)),jnp.array([timebound])])


def expsched(step1,timebound,doublingtime=4):

	df_over_f=jnp.log(2)/doublingtime
	transition=step1/df_over_f

	s_=[2**i for i in np.arange(math.floor(np.log(transition)/np.log(2)),np.log(timebound)/np.log(2),1/doublingtime)]
	s=list(np.arange(0,s_[0] if len(s_)>0 else timebound,step1))+s_
	return s+[timebound] if s[-1]<timebound else s
	


def periodicsched(step,timebound):
	s=list(np.arange(step,timebound,step))
	return s+[timebound] if len(s)==0 or s[-1]<timebound else s

def stepwiseperiodicsched(stepsizes,transitions):
	return jnp.concatenate([jnp.arange(transitions[i],transitions[i+1],step) for i,step in enumerate(stepsizes)]+[jnp.array([transitions[-1]])])



class Scheduler(Timer):
	def __init__(self,schedule):
		self.schedule=schedule
		self.rem_sched=deque(copy.deepcopy(schedule))
		Timer.__init__(self)

	def activate(self,t=None):
		if t==None:t=self.time()

		if len(self.rem_sched)==0 or t<self.rem_sched[0]:
			act=False
		else:
			while len(self.rem_sched)>0 and t>=self.rem_sched[0]:
				self.rem_sched.popleft()
			act=True

		return act

		
#	def filter(self,times,*valueshists):
#		timeticks=[times[0]]
#		filteredhists=[[hist[0]] for hist in valueshists]
#		try:
#			for t,*values in list(zip(times,*valueshists))[1:]:
#				if self.dispatch(t):
#					timeticks.append(t)
#					for val,hist in zip(values,filteredhists):
#						hist.append(val)
#		except Timeup:
#			pass
#		return (timeticks,*filteredhists)
#
#
#		#return (timeticks,)+filteredhists
#
#
#def filterschedule(sched,times,*valueshist):
#	sc=Scheduler(sched)
#	out=sc.filter(times,*valueshist)	
#	del sc
#	return out
#
#
#def filterschedule_w_ordinals(sched,times,*valshist):
#	_valshist=(list(range(len(times))),)+tuple(valshist)
#	return filterschedule(times,*_valshist)
#	
#
#def times_to_ordinals(allts,ticks,*vals):
#	ordn=0
#	ts=deque(allts)
#	ordns=[]
#
#	for tick in ticks:
#		while len(ts)>0 and ts[0]<tick:
#			ts.popleft()
#			ordn=ordn+1
#		ordns.append(ordn)
#	return (ordns,*vals)



#====================================================================================================


class Keychain:

	def __init__(self,seed=0):
		self.resetkeys(seed)

	def resetkeys(self,seed):
		self.keys=deque([rnd.PRNGKey(seed)])

	def refresh(self,key):
		_,*newkeys=rnd.split(key,1000)
		self.keys=deque(newkeys)

	def nextkey(self):
		if len(self.keys)==1: self.refresh(self.keys[-1])
		return self.keys.popleft()

keychain=Keychain()

def nextkey():
	return keychain.nextkey()




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
	return folder+max([(subfolder,relorder(subfolder)) for subfolder in os.listdir(folder)],key=lambda pair:pair[1])[0]+'/'
	
def latest(folder):
	folders=[f for f in os.listdir(folder) if len(f)==15 and len(re.sub('[^0-9]','',f))==10]
	def relorder(subfoldername):
		return int(re.sub('[^0-9]','',subfoldername))
	return folder+max([(subfolder,relorder(subfolder)) for subfolder in folders],key=lambda pair:pair[1])[0]+'/'







		


def getlossfn():
	return lossfn

#setlossfn('sqloss')
#setlossfn('SI_loss')
#setlossfn('log_SI_loss')




#lossfn=sqloss
heavy_threshold=8
BOX='\u2588'
box='\u2592'
dash='\u2015'

t0=time.perf_counter()
trackedvals=dict()
eventlisteners=dict()
sessionID=nowstr()
trackduration=False

biasinitsize=.1

hour=3600
day=24*hour
week=7*day

cmdparams,cmdredefs=parse_cmdln_args()









#sessionstate=State()
session=Memory()









# testing

if __name__=='__main__':

#	print(stepwiseperiodicsched([10,100],[0,60,600]))
#	for i in range(10):
#		print(nextkey())

	#print(selectone({'r','t'},[1,4,'r',5,'d']))

#	print(times_to_ordinals([.1,.2,.3,.4,.5,.6,.7,.8],[.3,.7],['a','b']))
#	print(times_to_ordinals([.1,.2,.3,.4,.5,.6,.7,.8],[.1,.2,.3,.4,.5,.6,.7,.8],[1,2,3,4,5,6,7,8]))

	print(expsched(.1,100,3))

