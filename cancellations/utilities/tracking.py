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


#----------------------------------------------------------------------------------------------------

class dotdict(dict):
    __getattr__=dict.get
    def __setattr__(self,k,v):
        self[k]=v

class Profile(dotdict):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)

    def butwith(self,**defs):
        self.update(defs)
        return self

    def __str__(self):
        return '\n'.join(['{} = {}'.format(k,v) for k,v in self.items()])

#----------------------------------------------------------------------------------------------------

class History:
    def __init__(self,membound=None):
        self.snapshots=deque()
        self.membound=membound

    def remember(self,val,**metadata):
        self.snapshots.append((val,metadata))
        if self.membound!=None and len(self.snapshots)>self.membound: self.snapshots.popleft()

    def gethist(self,*metaparams):
        valhist=[val for val,metadata in self.snapshots]
        metaparamshist=list(zip(*[[metadata[mp] for mp in metaparams] for val,metadata in self.snapshots]))

        return (valhist,*metaparamshist) if len(metaparams)>0 else valhist

    def getlastval(self):
        return self.snapshots[-1][0] if len(self.snapshots)>0 else None


#    def filter(self,filterby,schedule):
#        filteredhistory=History()
#        schedule=deque(schedule)
#
#        for val,metadata in self.snapshots:
#            t=metadata[filterby]
#            if t>=schedule[0]:
#                filteredhistory.remember(val,metadata)
#                while t>=schedule[0]:
#                    schedule.popleft()
#                    if len(schedule)==0: break
#                if len(schedule)==0: break
#        return filteredhistory

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
        if membound!=None: self.hists[name].membound=membound
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


class Watched:
    def __init__(self):
        self.signals=dict()

    def addlistener(self,listener,signal):
        if signal not in self.signals: self.signals[signal]=[]
        self.signals[signal].append(listener)

    def pokelisteners(self,signal):
        if signal in self.signals:
            for listener in self.signals[signal]: listener.poke(self)

#
#class Memory(BasicMemory,Timer,Watched):
#    def __init__(self):
#        BasicMemory.__init__(self)
#        Timer.__init__(self)
#        Watched.__init__(self)
#        self.memID=random.randint(0,10**9)
#        self.context=dict()
#
#    def addcontext(self,name,val):
#        self.context[name]=val
#        self.remember(name,val)
#        self[name]=val
#
#    def getcontext(self):
#        return self.context|{'memory {} time'.format(self.memID):self.time()}
#
#    def gethistbytime(self,name):
#        timename='memory {} time'.format(self.memID)
#        out=self.gethist(name,timename)
#        return out
#
#    def remember(self,varname,val,**context):
#        super().remember(varname,val,self.getcontext()|context)
#        #self.pokelisteners(varname)
#        #poke(varname,val)
#

#----------------------------------------------------------------------------------------------------

class RunningAvg:
    def __init__(self,k):
        self.k=k
        self.recenthist=deque([])
        self._sum_=0
        self._sqsum_=0
        self.i=0

    def update(self,val,thinning=1):
        if self.i%thinning==0: self.do_update(val)
        return self.avg()

    def do_update(self,val):    
        self.i+=1
        self._sum_+=val
        self._sqsum_+=val**2
        self.recenthist.append(val)
        if len(self.recenthist)>self.k:
            self._sum_-=self.recenthist.popleft()

    def avg(self):
        return self.sum()/self.actualk()    

    def var(self,val=None,**kw):
        if val!=None: self.update(val,**kw)
        return self.sqsum()/self.actualk()-self.avg()**2

    def actualk(self):
        return len(self.recenthist)

    def sum(self): return self._sum_
    def sqsum(self): return self._sqsum_

class InfiniteRunningAvg(RunningAvg):
    def __init__(self):
        self._sum_=0
        self._sqsum_=0
        self.i=0

    def do_update(self,val):    
        self.i+=1
        self._sum_+=val
        self._sqsum_+=val**2

    def actualk(self): return self.i


def ispoweroftwo(n):
    pattern=re.compile('10*')
    return pattern.fullmatch('{0:b}'.format(n))


class ExpRunningAverage(InfiniteRunningAvg):
    def __init__(self):
        self.blocksums=[]
        self.intervals=[]
        self.i=0

    def do_update(self,val):
        if ispoweroftwo(self.i) or self.i==0:
            self.blocksums.append(InfiniteRunningAvg())
            self.intervals.append([self.i,self.i])
        self.blocksums[-1].do_update(val)
        self.intervals[-1][-1]+=1
        self.i+=1

    def sum(self):
        return sum([e.sum() for e in self.blocksums])

    def sqsum(self): return sum([e.sqsum() for e in self.blocksums])

    def avg(self):
        if self.i<=1: return None
        prevlen=self.intervals[-2][1]-self.intervals[-2][0]
        curlen=self.intervals[-1][1]-self.intervals[-1][0]
        return (self.blocksums[-1].sum()+self.blocksums[-2].sum())/(prevlen+curlen)






class NoRunningAvg(RunningAvg):
    def __init__(self): pass
    def update(self,val): self.val=val; return val
    def avg(self): return self.val
    def actualk(self): return 1


def RunningAvgOrIden(k):
    if k==1: return NoRunningAvg()
    if k==None: return InfiniteRunningAvg()
    return RunningAvg(k)

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

#def donothing(*x,**y):
#    return None




class Process(Memory):
    def __init__(self,profile=None,**kw):
        super().__init__()

        assert(profile==None or len(kw)==0)
        if profile==None: profile=Profile(profilename='emptyprofile',**kw)

        self.keychain=Keychain()
        self.profile=profile
        self.setID()
        self.outpath='outputs/'+self.ID+'/'

    def setID(self):
        self.ID='{}/{}/{}'.format(self.processname,self.profile.profilename,setup.session.ID)

    def log(self,msg):
        msg=str(msg)
        tmsg=textutil.appendright(self.timeprint()+' | ',msg)
        self.remember('recentlog',tmsg,membound=100)
        sysutil.write(tmsg,self.outpath+'log.txt',mode='a')

        self.refresh()

    def nextkey(self):
        return self.keychain.nextkey()

    def refresh(self): pass


#    def logcurrenttask(self,msg):
#        self.trackcurrenttask(msg,0)
#        log(msg)
#
#    def trackcurrenttask(self,msg,completeness):
#        if completeness>=1 or stopwatch.tick_after(.05):
#            self.currenttask=msg
#            self.currenttaskcompleteness=completeness
#            return act_on_input(setup.checkforinput())
#        else: return None
#
#    def getcurrenttask(self):
#        try: return self.run.getval('currenttask')	
#        except: None
#
#    def clearcurrenttask(self):
#        self.currenttask=None
#        self.currenttaskcompleteness=0
#
#    @staticmethod
#    def getdefaultprofile():
#        return Profile()

    #def profilestr(self):
    #    return '\n'.join(['{} = {}'.format(k,v) for k,v in self.profile.items()])





class Session(Process):
    processname='session'

    def setID(self):
        self.ID=nowstr()


def nowstr():
    date,time=str(datetime.datetime.now()).split('.')[0].split(' ')
    date='-'.join(date.split('-')[1:])
    time=''.join([x for pair in zip(time.split(':'),['h','m','s']) for x in pair])
    return date+'|'+time


#def newprocess(Processtype,**kw):
#    return Processtype(Profile(**kw))


#class Run(Process): pass
#    def __init__(self,*a,**kw):
#        super().__init__(*a,**kw)
#        self.X_distr=lambda key,samples: self.profile._X_distr_(key,samples,self.profile.n,self.profile.d)
#
#    def genX(self,samples:int):
#        return self.X_distr(self.nextkey(),samples)




#
#
#def newprocess(execfn):
#    class CustomProcess(Process):
#        execprocess=execfn
#    return CustomProcess
#
#


#def log(msg):
#    session.log(msg)
#    sysutil.write(msg+'\n',currentprocess().outpath+'log')
#    return currentprocess().profile.act_on_input(setup.checkforinput())


#def LOG(msg):
#    log('\n\n'+msg+'\n\n')
#
#----------------------------------------------------------------------------------------------------



#def dblog(msg):
#    write(str(msg)+'\n','dblog/'+sessionID)
#    write(str(msg)+'\n\n',outpath+'dblog')


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
#
#class Breaker:
#    def __init__(self):
#        self.wantbreak=False
#
#    def breaknow(self):
#        self.wantbreak=True
#
#    def wantsbreak(self):
#        out=self.wantbreak
#        self.wantbreak=False
#        return out
#
#
#breaker=Breaker()
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
        if t==None:t=self.time()

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
setup.session=Session()

def log(msg):
    setup.session.log(msg)

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
    if len(processes)>0 and (process==None or processes[-1]==process):
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




#def pull(*varnames):
#    process=currentprocess()
#    return [process[vn] for vn in varnames]

def act_on_input(inp):
    return currentprocess().profile.act_on_input(inp)



#----------------------------------------------------------------------------------------------------
def nextkey(): return currentprocess().nextkey()
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
#	#sessionstate.save(*paths)
#		
#def save():
#	savestate(*histpaths())
#

#def formatvars(elements,separator=' ',ignore={}):
#    return separator.join(['{}={}'.format(name,val) for name,val in elements if name not in ignore])


#----------------------------------------------------------------------------------------------------





def providedefault(defs,**kw):
    [(name,defaultval)]=list(kw.items())
    try: return defs[name]
    except: return defaultval


t0=time.perf_counter()
trackedvals=dict()
eventlisteners=dict()




def getfromargs(**kw):
    return kw[selectone(set(kw.keys()),cmdparams)]

fromcmdparams=getfromargs
getfromcmdparams=getfromargs


dash='\u2015'









def test():
    import time
    s=Stopwatch()
    for i in range(100): print(s.tick_after(.1)); time.sleep(.01)


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
#	if 'recentlog' in args:
#		print(args[1])




    #try: return chr(a)
    #except: return a
    #try: return {259:'UP',258:'DOWN',260:'LEFT',261:'RIGHT'}[a]
    #except: pass
    #try: return {27:'ESCAPE',127:'BACKSPACE',10:'ENTER'}[a]
    #except: pass
    #return a




#def printonpoke(msgfn):
#	def newpoke(*args,**kw):
#		print(msgfn(*args,**kw))
#	global poke
#	poke=newpoke
#
#def print_task_on_poke():
#	def msgfn(*args,**kw):
#		return '{}: {:.0%}'.format(session.getcurrentval('currenttask'),\
#			session.getcurrentval('currenttaskcompleteness'))
#	printonpoke(msgfn)


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
##	print(stepwiseperiodicsched([10,100],[0,60,600]))
##	for i in range(10):
##		print(nextkey())
#
#    #print(selectone({'r','t'},[1,4,'r',5,'d']))
#
##	print(times_to_ordinals([.1,.2,.3,.4,.5,.6,.7,.8],[.3,.7],['a','b']))
##	print(times_to_ordinals([.1,.2,.3,.4,.5,.6,.7,.8],[.1,.2,.3,.4,.5,.6,.7,.8],[1,2,3,4,5,6,7,8]))
#
#    #print(expsched(.1,100,3))
#
#
#    print(nonsparsesched(1000,10))
#
#    #livekeyboard()
#