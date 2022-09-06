import numpy as np
import math
import pickle
import time
import copy
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import datetime
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
		#self.remember('log',msg,{'timeprint':tp})
		self.remember('recentlog',msg,membound=25)

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

class Watched:
	def __init__(self):
		self.signals=dict()

	def addlistener(self,listener,signal):
		if signal not in self.signals: self.signals[signal]=[]
		self.signals[signal].append(listener)

	def pokelisteners(self,signal):
		if signal in self.signals:
			for listener in self.signals[signal]: listener.poke(self)



class Memory(BasicMemory,Timer,Watched):
	def __init__(self):
		BasicMemory.__init__(self)
		Timer.__init__(self)
		Watched.__init__(self)
		self.memID=int(rnd.randint(nextkey(),minval=0,maxval=10**9,shape=(1,)))
		self.context=dict()

	def addcontext(self,name,val):
		self.context[name]=val
		self.trackcurrent(name,val)

	def getcontext(self):
		return self.context|{'memory {} time'.format(self.memID):self.time()}

	def gethistbytime(self,name):
		timename='memory {} time'.format(self.memID)
		out=self.gethist(name,timename)
		return out

	def remember(self,varname,val,**context):
		super().remember(varname,val,self.getcontext()|context)
		self.pokelisteners(varname)
		#poke(varname,val)



class ActiveMemory(Memory):

	def compute(self,queries,fn,outputname):
		inputvals=[self.getcurrentval(q) for q in queries]
		self.remember(outputname,fn(*inputvals))

	def computefromhist(self,queries,fn,outputname):
		inputvals=[self.gethist(q) for q in queries]
		self.remember(outputname,fn(*inputvals))


def timeprint():
	return datetime.timedelta(seconds=int(session.time()))

class Session(Memory):
	def log(self,msg):
		msg='{} | {}'.format(timeprint(),msg)
		write(msg+'\n',*logpaths())
		if trackduration:
			write(str(int(session.time())),*[os.sep.join(pathlog.split(os.sep)[:-1])+os.sep+'duration' for pathlog in logpaths()],mode='w')	
		super().log(msg)
		if rawlogprint: print(msg)

rawlogprint=True

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

def getoutpath():
	return outpath

def register(_dict_,names):
	params.update({k:_dict_[k] for k in names})

#
#def savestate(*paths):
#	#sessionstate.save(*paths)
#		
#def save():
#	savestate(*histpaths())
#



#----------------------------------------------------------------------------------------------------

def log(msg):
	session.log(msg)
	return act_on_input(getinput())

def dblog(msg):
	write(str(msg)+'\n','dblog/'+sessionID)



class Stopwatch:
	def __init__(self):
		self.t=time.perf_counter()

	def elapsed(self):
		return time.perf_counter()-self.t

	def tick(self):
		t0=self.t
		self.t=time.perf_counter()
		return self.t-t0

	def tick_after(self,dt):
		if self.elapsed()>dt:
			self.tick()
			return True
		else:
			return False

	def dbtick(self,msg=''):
		dbprint('{} {}'.format(msg,self.tick()))



#----------------------------------------------------------------------------------------------------




# start must be 10,25,50,100,250,500,1000,...
def sparsesched(timebound,start=500,skipzero=True):
	sched=[0]
	t=start
	while t<timebound:
		sched.append(t)
		t=intuitive_exp_increment(t)
	sched.append(timebound)
	return sched[1:] if skipzero else sched

# start must be 10,25,50,100,250,500,1000,...
def nonsparsesched(timebound,start=10,skipzero=False):
	sched=[]
	t=0
	dt=start
	while t<timebound:
		sched.append(t)
		t+=dt
		if t>=10*dt:
			dt=intuitive_exp_increment(dt)
			
	sched.append(timebound)
	return sched[1:] if skipzero else sched


# t must be 10,25,50,100,250,500,1000,...
def intuitive_exp_increment(t):
	if str(t)[0]=='1':
		return int(t*2.5) if t>=10 else t*2.5
	else:
		return int(t*2)


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

		

#====================================================================================================

class Clockedworker(Stopwatch):

	def __init__(self):
		super().__init__()
		self.totalrest=0
		self.totalwork=0
		self.working=False

	def clock_in(self):
		assert not self.working
		self.totalrest+=self.tick()
		self.working=True

	def clock_out(self):
		assert self.working
		self.totalwork+=self.tick()
		self.working=False

	def workfraction(self):
		return self.totalwork/self.elapsed()
	
	def do_if_rested(self,workfraction,*fs):
		if self.workfraction()<workfraction:
			self.clock_in()
			for f in fs:
				f()
			self.clock_out()		


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

keychains={s:Keychain(s) for s in range(10)}|{'weights':Keychain(10)}

def nextkey(c=None):
	if c==None: c=currentkeychain
	return keychains[c].nextkey()

currentkeychain=0


#====================================================================================================

def makedirs(filepath):
	path='/'.join(filepath.split('/')[:-1])
	filename=filepath.split('/')[-1]
	os.makedirs(path,exist_ok=True)	

def save(data,*paths,echo=True):
	for path in paths:
		makedirs(path)
		with open(path,'wb') as file:
			pickle.dump(data,file)
	if echo: log('Saved data to {}'.format(paths))

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

		
def showfile(path):
	import os
	import subprocess
	log('opening path '+path)

	try: subprocess.Popen(['open',path])
	except: pass
	try: subprocess.Popen(['xdg-open',path])
	except: pass
	try: os.startfile(path)
	except: pass

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

def selectonefromargs(*options):
	return selectone(set(options),parse_cmdln_args()[0])

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






def memorybatchlimit(n):
	s=1
	#memlim=50000
	memlim=10000
	while(s*math.factorial(n)<memlim):
		s=s*2

	if n>heavy_threshold:
		assert s==1, 'AS_HEAVY assumes single samples'

	return s
		


def getlossfn():
	return lossfn

#setlossfn('sqloss')
#setlossfn('SI_loss')
#setlossfn('log_SI_loss')

def genX(samples):
	return X_distr(nextkey(),samples)


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

def getfromargs(**kw):
	return kw[selectone(set(kw.keys()),cmdparams)]

fromcmdparams=getfromargs
getfromcmdparams=getfromargs


def getinput():
	return None

def act_on_input(c):
	return c

#sessionstate=State()
session=Session()
params=dict()

#def poke(*args,**kw):
#	if 'recentlog' in args:
#		print(args[1])


def donothing(*x,**y):
	return None

def conditional(f,do):
	if do:
		f()


def extractkey_cs(a):
	try: return chr(a)
	except: pass
	try: return {259:'UP',258:'DOWN',260:'LEFT',261:'RIGHT'}[a]
	except: pass
	try: return {27:'ESCAPE',127:'BACKSPACE',10:'ENTER'}[a]
	except: pass
	return a




def logcurrenttask(msg):
	trackcurrenttask(msg,0)
	log(msg)

def trackcurrenttask(msg,completeness):
	session.trackcurrent('currenttask',msg)
	session.trackcurrent('currenttaskcompleteness',completeness)
	return act_on_input(getinput())
	

def clearcurrenttask():
	session.trackcurrent('currenttask',' ')
	session.trackcurrent('currenttaskcompleteness',0)



def printonpoke(msgfn):
	def newpoke(*args,**kw):
		print(msgfn(*args,**kw))
	global poke
	poke=newpoke

def print_task_on_poke():
	def msgfn(*args,**kw):
		return '{}: {:.0%}'.format(session.getcurrentval('currenttask'),\
			session.getcurrentval('currenttaskcompleteness'))
	printonpoke(msgfn)

def indent(s):
	return '\n'.join(['    '+l for l in s.splitlines()])

def provide(**kw):
	for name,val in kw.items():
		if name not in globals():
			globals()[name]=val	

def addparams(**kw):
	params.update(kw)


plotfineness=50

dash='\u2015'

def wraptext(msg,style=dash):
	width=max([len(l) for l in msg.splitlines()])
	line=dash*width
	return '{}\n{}\n{}'.format(line,msg,line)

def wraplines(lines,style=dash):
	width=max([len(l) for l in lines])
	line=dash*width
	return [line]+lines+[line]

####################################################################################################

# testing


if __name__=='__main__':

#	print(stepwiseperiodicsched([10,100],[0,60,600]))
#	for i in range(10):
#		print(nextkey())

	#print(selectone({'r','t'},[1,4,'r',5,'d']))

#	print(times_to_ordinals([.1,.2,.3,.4,.5,.6,.7,.8],[.3,.7],['a','b']))
#	print(times_to_ordinals([.1,.2,.3,.4,.5,.6,.7,.8],[.1,.2,.3,.4,.5,.6,.7,.8],[1,2,3,4,5,6,7,8]))

	#print(expsched(.1,100,3))


	print(nonsparsesched(1000,10))

	#livekeyboard()
