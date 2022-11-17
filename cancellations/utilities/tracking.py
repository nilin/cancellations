import numpy as np
import time
import copy
import jax.numpy as jnp
import sys
from . import sysutil,config as cfg, textutil, setup
import matplotlib.pyplot as plt
import datetime
import re
import jax.random as rnd
import copy
from collections import deque
import datetime
import random
import copy

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
    return zip(*[(snapshot[n] for n in varnames) for snapshot in snapshots if all([n in snapshot.keys() for n in varnames])])



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



class Process(Memory):
    processname='process'

    def __init__(self,profile=None,**kw):
        super().__init__()

        assert(profile is None or len(kw)==0)
        if profile is None:
            try: profile=self.getdefaultprofile(**kw)
            except: profile=self.getdefaultprofile().butwith(**kw)

        self.keychain=Keychain()
        self.profile=profile
        self.setID()
        self.outpath='outputs/'+self.ID+'/'

        self.continueprocess=self.execprocess

    def setID(self):
        self.ID='{}/{}/{}'.format(self.processname,self.profile.profilename,setup.session.ID)

    def log(self,msg):
        msg=str(msg)
        tmsg=textutil.appendright(self.timeprint()+' | ',msg)
        self.remember('recentlog',tmsg,membound=100)
        sysutil.write(tmsg+'\n',self.outpath+'log.txt',mode='a')

        self.refresh()

        if setup.display_on==False: print(msg)

    def nextkey(self):
        return self.keychain.nextkey()

    def refresh(self): pass

    @staticmethod
    def getdefaultprofile(**kw):
        return Profile().butwith(**kw)

    @classmethod
    def getprofiles(cls,**kw):
        return cls.getdefaultprofile(**kw)



def nowstr():
    date,time=str(datetime.datetime.now()).split('.')[0].split(' ')
    date='-'.join(date.split('-')[1:])
    time=''.join([x for pair in zip(time.split(':'),['h','m','s']) for x in pair])
    return date+'|'+time


class Session(Process):
    processname='session'
    def __init__(self,profile=None,**kw):
        super().__init__(profile=profile,**kw)
        self.outpath='outputs/sessions/'+self.ID+'/'

    def setID(self):
        self.ID=nowstr()


setup.session=Session()

def log(msg):
    currentprocess().log(msg)
    #setup.session.log(msg)

def getlog():
    return setup.session.gethist('recentlog')




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



loadprocess(setup.session)

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
# 
# class Clockedworker(Stopwatch):
# 
#     def __init__(self):
#         super().__init__()
#         self.totalrest=0
#         self.totalwork=0
#         self.working=False
# 
#     def clock_in(self):
#         assert not self.working
#         self.totalrest+=self.tick()
#         self.working=True
# 
#     def clock_out(self):
#         assert self.working
#         self.totalwork+=self.tick()
#         self.working=False
# 
#     def workfraction(self):
#         return self.totalwork/self.elapsed()
#     
#     def do_if_rested(self,workfraction,*fs):
#         if self.workfraction()<workfraction:
#             self.clock_in()
#             for f in fs:
#                 f()
#             self.clock_out()        
# 
# 
#====================================================================================================


setup.stopwatch=Stopwatch()
# def logcurrenttask(msg): currentprocess().logcurrenttask(msg)
# def trackcurrenttask(msg,completeness): return currentprocess().trackcurrenttask(msg,completeness)
# def getcurrenttask(): return currentprocess().getcurrenttask()
# def clearcurrenttask(): currentprocess().clearcurrenttask()
#----------------------------------------------------------------------------------------------------


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




#def getlossfn():
#    return lossfn

#def histpath():
#    return cfg.outpath+'hist'
#
#def logpaths():
#    return [cfg.outpath+'log','logs/'+session.ID]
#
#def getoutpath():
#    return cfg.outpath

#def register(*names,sourcedict,savetoglobals=False):
#    cfgcontext=globals() if savetoglobals else params
#    cfgcontext.update({k:sourcedict[k] for k in names})
#
#def retrieve(context,names):
#    context.update({k:params[k] if k in params else globals()[k] for k in names})


#
#def savestate(*paths):
#    #sessionstate.save(*paths)
#        
#def save():
#    savestate(*histpaths())
#

#def formatvars(elements,separator=' ',ignore={}):
#    return separator.join(['{}={}'.format(name,val) for name,val in elements if name not in ignore])


#----------------------------------------------------------------------------------------------------


#
#
#
#def providedefault(defs,**kw):
#    [(name,defaultval)]=list(kw.items())
#    try: return defs[name]
#    except: return defaultval
#
#
#t0=time.perf_counter()
#trackedvals=dict()
#eventlisteners=dict()
#
#
#
#
#def getfromargs(**kw):
#    return kw[selectone(set(kw.keys()),cmdparams)]
#
#fromcmdparams=getfromargs
#getfromcmdparams=getfromargs
#
#
#dash='\u2015'
#
#


#
#
#
#
#
#def test():
#    import time
#    s=Stopwatch()
#    for i in range(100): print(s.tick_after(.1)); time.sleep(.01)
#
#
#def conditional(f,do):
#    if do:
#        f()

#def orderedunion(A,B):
#    A,B=list(A),deque(B)
#    S=set(A)
#    for b in B:
#        if b not in S:
#            A.append(b)
#            S.add(b)
#    return A
#        
#
#def terse(l):
#    return str([round(float(e*1000))/1000 for e in l])
#
#def selectone(options,l):
#    choice=list(options.intersection(l))
#    assert(len(choice)==1)
#    return choice[0]
#
#def selectonefromargs(*options):
#    return selectone(set(options),parse_cmdln_args()[0])

#====================================================================================================


#def longestduration(folder):
#    def relorder(subfolder):
#        try:
#            with open(folder+'/'+subfolder+'/duration','r') as f:
#                return int(f.read())
#        except:
#            return -1
#    return folder+max([(subfolder,relorder(subfolder)) for subfolder in os.listdir(folder)],key=lambda pair:pair[1])[0]+'/'
#    
#def latest(folder):
#    folders=[f for f in os.listdir(folder) if len(f)==15 and len(re.sub('[^0-9]','',f))==10]
#    def relorder(subfoldername):
#        return int(re.sub('[^0-9]','',subfoldername))
#    return folder+max([(subfolder,relorder(subfolder)) for subfolder in folders],key=lambda pair:pair[1])[0]+'/'


#def memorybatchlimit(n):
#    s=1
#    memlim=10**6
#    while(s*math.factorial(n)<memlim):
#        s=s*2
#
#    if n>heavy_threshold:
#        assert s==1, 'AS_HEAVY assumes single samples'
#
#    return s
        



#setlossfn('sqloss')
#setlossfn('SI_loss')
#setlossfn('log_SI_loss')



#lossfn=sqloss
#heavy_threshold=8


#sessionstate=State()
#params=dict()

#def poke(*args,**kw):
#    if 'recentlog' in args:
#        print(args[1])




    #try: return chr(a)
    #except: return a
    #try: return {259:'UP',258:'DOWN',260:'LEFT',261:'RIGHT'}[a]
    #except: pass
    #try: return {27:'ESCAPE',127:'BACKSPACE',10:'ENTER'}[a]
    #except: pass
    #return a




#def printonpoke(msgfn):
#    def newpoke(*args,**kw):
#        print(msgfn(*args,**kw))
#    global poke
#    poke=newpoke
#
#def print_task_on_poke():
#    def msgfn(*args,**kw):
#        return '{}: {:.0%}'.format(session.getcurrentval('currenttask'),\
#            session.getcurrentval('currenttaskcompleteness'))
#    printonpoke(msgfn)


#def provide(context=None,**kw):
#    if context==None: context=globals()
#    for name,val in kw.items():
#        if name not in context:
#            context[name]=val    
#


#def addparams(**kw):
#    params.update(kw)





#def refreshdisplay(name):
#    try: dashboard.draw(name)
#    except: log('failed to refresh display '+name)


####################################################################################################

# testing


#checkforinput=donothing



#
#
#
#
#
#if __name__=='__main__':
#
##    print(stepwiseperiodicsched([10,100],[0,60,600]))
##    for i in range(10):
##        print(nextkey())
#
#    #print(selectone({'r','t'},[1,4,'r',5,'d']))
#
##    print(times_to_ordinals([.1,.2,.3,.4,.5,.6,.7,.8],[.3,.7],['a','b']))
##    print(times_to_ordinals([.1,.2,.3,.4,.5,.6,.7,.8],[.1,.2,.3,.4,.5,.6,.7,.8],[1,2,3,4,5,6,7,8]))
#
#    #print(expsched(.1,100,3))
#
#
#    print(nonsparsesched(1000,10))
#
#    #livekeyboard()
#