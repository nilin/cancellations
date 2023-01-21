import time
import datetime
import jax.random as rnd
from collections import deque
import datetime
import os
import copy
from functools import partial
from cancellations.config import sysutil, config as cfg
from cancellations.utilities import textutil
import jax.numpy as jnp

#----------------------------------------------------------------------------------------------------


processes=[]


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
    date=''.join(date.split('-')[1:])
    time=''.join([x for pair in zip(time.split(':'),['','','']) for x in pair])
    return date+'-'+time


class Process(Memory):
    processname='process'
    processtype='processes'

    def __init__(self,*a,profile=None,**kw):
        super().__init__()

        if profile is not None:
            self.profile=profile
        else:
            self.profile=self.getprofile(*a,**kw)

        self.keychain=Keychain()
        self.weapons=dict()

    def nextkey(self):
        return self.keychain.nextkey()

    def refresh(self): pass

    @classmethod
    def getprofile(cls,**kw):
        return Profile(**kw)




def log(*msgs,multiline=False,**kw):
    if multiline:
        msg='\n\n'.join([str(s) for s in msgs])
        tmsg=sessiontimer.timeprint()+'\n\n'+msg+'\n\n'
    else:
        msg=' '.join([str(s) for s in msgs])
        tmsg=textutil.appendright(sessiontimer.timeprint()+' | ',msg)

    sysutil.write(tmsg+'\n',logpath,mode='a')

    if cfg.display_on==False: print(msg)
    try:
        #breakpoint()
        #global maindash
        #maindash.draw()
        #getch()
        screen.refresh()
        #cfg.currentlogdisplay.draw()
        #cfg.getch()
        #cfg.screen.refresh()
    except Exception as e: 
        pass


#====================================================================================================




def loadprocess(process):
    if cfg.display_on:
        process.dashboard=currentprocess().dashboard.blankclone()
    processes.append(process)
    return process

def unloadprocess(process=None):
    if len(processes)>0 and (process is None or processes[-1]==process):
        if cfg.display_on:
            clearcurrentdash()
        process=processes.pop()
    return process

def runprocess(process,*a,**kw):
    loadprocess(process)
    out=process.execprocess(*a,**kw)
    unloadprocess(process)
    return out

def swap_process(process):
    unloadprocess()
    return loadprocess(process)

def currentprocess():
    return processes[-1]

def act_on_input(inp,*args,**kw):
    return currentprocess().profile.act_on_input(inp,*args,**kw)

def nextkey(): return currentprocess().nextkey()



#----------------------------------------------------------------------------------------------------







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



def periodicsched(step,timebound):
    s=list(np.arange(step,timebound,step))
    return s+[timebound] if len(s)==0 or s[-1]<timebound else s


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




session=Process()
processes=[session]

sessionID=nowstr()
sessiontimer=Timer()
logpath=os.path.join('logs',sessionID)

timedistribution=TimeDistribution()
stopwatch=Stopwatch()

hour=3600
day=24*hour
week=7*day