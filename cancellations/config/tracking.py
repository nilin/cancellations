import time
import datetime
import jax.random as rnd
from collections import deque
import datetime
import os
import copy
from cancellations.config import sysutil, config as cfg
from cancellations.utilities import textutil
import jax.numpy as jnp

#----------------------------------------------------------------------------------------------------

class dotdict(dict):
    __getattr__=dict.get
    def __setattr__(self,k,v):
        self[k]=v

    def __str__(self):
        return '\n'.join([textutil.appendright(k+' = ',str(v)) for k,v in self.items()])

class Profile(dotdict):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        if not hasattr(self,'profilename'):
            self.profilename='default'

    def butwith(self,**defs):
        newself=Profile(**self)
        newself.update(defs)
        return newself


#----------------------------------------------------------------------------------------------------


def extracthist(snapshots,*varnames):
    out=[[] for n in varnames]
    for snapshot in snapshots:
        if all([n in snapshot.keys() for n in varnames]):
            for i,n in enumerate(varnames):
                out[i].append(snapshot[n])
    return out
    #return zip(*[(snapshot[n] for n in varnames) for snapshot in snapshots if all([n in snapshot.keys() for n in varnames])])



#----------------------------------------------------------------------------------------------------

class History:
    def __init__(self,membound=None):
        self.snapshots=deque()
        self.membound=membound

    def remember(self,val,**metadata):
        self.snapshots.append((val,metadata))
        if self.membound is not None and len(self.snapshots)>self.membound: self.snapshots.popleft()

    def gethist(self,*metaparams):
        valhist=[val for val,metadata in self.snapshots]
        metaparamshist=list(zip(*[[metadata[mp] for mp in metaparams] for val,metadata in self.snapshots]))

        return (valhist,*metaparamshist) if len(metaparams)>0 else valhist

    def getlastval(self):
        return self.snapshots[-1][0] if len(self.snapshots)>0 else None



class Timer:
    def __init__(self,*x,**y):
        self.t0=time.perf_counter()

    def time(self):
        return time.perf_counter()-self.t0

    def timeprint(self):
        return str(datetime.timedelta(seconds=int(self.time())))

class Memory(dotdict,Timer):
    def __init__(self):
        self.hists=dict()
        Timer.__init__(self)

    def remember(self,name,val,membound=None,**kw):
        if name not in self.hists:
            self.hists[name]=History(membound=membound)
        if membound is not None: self.hists[name].membound=membound
        self.hists[name].remember(val,**kw)

    def gethist(self,name,*metaparams):
        return self.hists[name].gethist(*metaparams) if name in self.hists else\
            (([],)+tuple([[] for _ in metaparams]) if len(metaparams)>0 else [])

    def getval(self,name):
        return self.hists[name].getlastval() if name in self.hists else self[name]
#
    def getqueryfn(self,*names):
        if len(names==1):
            [name]=names
            return lambda: self.getval(name)
        else:
            return lambda: [self.getval(name) for name in names]

#----------------------------------------------------------------------------------------------------

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

#----------------------------------------------------------------------------------------------------

def nowstr():
    date,time=str(datetime.datetime.now()).split('.')[0].split(' ')
    date='-'.join(date.split('-')[1:])
    time=''.join([x for pair in zip(time.split(':'),['h','m','s']) for x in pair])
    return date+'|'+time


class Process(Memory):
    processname='process'
    processtype='processes'

    def __init__(self,profile=None,**kw):
        super().__init__()

        assert(profile is None or len(kw)==0)
        if profile is None:
            try: profile=self.getdefaultprofile(**kw)
            except: profile=self.getdefaultprofile().butwith(**kw)

        self.keychain=Keychain()
        self.profile=profile
        self.setID()
        self.outpath=os.path.join('outputs',self.processtype,self.ID)
        self.continueprocess=self.execprocess

    def setID(self):
        self.ID=cfg.session.ID
        #self.ID=nowstr()
        #self.ID='{}/{}/{}'.format(self.processname,self.profile.profilename,cfg.session.ID)

    def log(self,msg):
        msg=str(msg)
        tmsg=textutil.appendright(self.timeprint()+' | ',msg)
        self.remember('recentlog',tmsg,membound=100)
        sysutil.write(tmsg+'\n',os.path.join(self.outpath,'log.txt'),mode='a')

        self.refresh()

        if cfg.display_on==False: print(msg)

    def nextkey(self):
        return self.keychain.nextkey()

    def refresh(self): pass

    @staticmethod
    def getdefaultprofile(**kw):
        return Profile().butwith(**kw)

    @classmethod
    def getprofiles(cls,**kw):
        return cls.getdefaultprofile(**kw)





class Session(Process):
    processname='session'
    processtype='sessions'

    def setID(self):
        self.ID=nowstr()


cfg.session=Session()
cfg.log=cfg.session.log


def log(msg):
    currentprocess().log(msg)

def getlog():
    return cfg.session.gethist('recentlog')




#====================================================================================================

processes=[]
dashboards=[]



def loadprocess(process):
    if len(processes)==0 or processes[-1]!=process:
        processes.append(process)
        dashboards.append(dict())
    return process

def unloadprocess(process=None):
    if len(processes)>0 and (process is None or processes[-1]==process):
        processes.pop()
        dashboards.pop()
    return process

def swap_process(process):
    unloadprocess()
    return loadprocess(process)


def currentprocess():
    return processes[-1]

def currentdashboard():
    return dashboards[-1]


def act_on_input(inp,*args,**kw):
    return currentprocess().profile.act_on_input(inp,*args,**kw)


def nextkey(): return currentprocess().nextkey()


loadprocess(cfg.session)

#----------------------------------------------------------------------------------------------------






class Pointer(dotdict): pass
    


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
        if self.elapsed()>=dt:
            self.tick()
            return True
        else:
            return False

    def do_after(self,dt,fn,*args,**kwargs):
        if self.tick_after(dt):
            fn(*args,**kwargs)

#
#----------------------------------------------------------------------------------------------------




# start must be 10,25,50,100,250,500,1000,...
def sparsesched(timebound,start=500,skipzero=False):
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
        if t is None:t=self.time()

        if len(self.rem_sched)==0 or t<self.rem_sched[0]:
            act=False
        else:
            while len(self.rem_sched)>0 and t>=self.rem_sched[0]:
                self.rem_sched.popleft()
            act=True

        return act

        

#====================================================================================================


class TimeDistribution:

    none='everything else'

    def __init__(self,log_every=10):
        self.trackblocks=Stopwatch()
        self.updatetracker=Stopwatch()
        self.current=self.none
        self.times={self.none:0}
        self.log_every=log_every
    
    def starttask(self,task):
        self.concludecurrenttask()
        self.current=task
        if task not in self.times:
            self.times[task]=0
            self.times[self.none]=self.times.pop(self.none)

    def concludecurrenttask(self):
        task=self.current
        self.times[task]+=self.trackblocks.tick()
        if self.log_every is not None:
            self.updatetracker.do_after(self.log_every,self.log)

    def endtask(self):
        self.starttask(self.none)

    def log(self):
        tasks,times=zip(*list(self.times.items()))
        times=jnp.array(times)
        times=times/jnp.sum(times)
        for task,time in zip(tasks,times):
            log('{:.0%} of time spent on {}'.format(time,task))

cfg.timedistribution=TimeDistribution()
cfg.stopwatch=Stopwatch()

def debuglistcomp(x,msg):
    log(msg)
    return x

def testX(n=5,d=2):
    return rnd.normal(nextkey(),(100,n,d))

BOX='\u2588'
box='\u2592'
dash='\u2015'
hour=3600
day=24*hour
week=7*day

