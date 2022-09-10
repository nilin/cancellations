from . import util,sysutil
import sys
import time
from collections import deque

processes=[]




def loadprocess(process):
    if len(processes)==0 or processes[-1]!=process:
        processes.append(process)

def unloadprocess(process):
    if len(processes)>0 and processes[-1]==process:
        processes.pop()

def currentprocess():
    return processes[-1]

def pull(*varnames):
    process=currentprocess()
    return [process[vn] for vn in varnames]

def act_on_input(inp):
    return processes[-1].act_on_input(inp)



#----------------------------------------------------------------------------------------------------
def nextkey(): return currentprocess().nextkey()
def logcurrenttask(msg): currentprocess().logcurrenttask(msg)
def trackcurrenttask(msg,completeness): return currentprocess().trackcurrenttask(msg,completeness)
def getcurrenttask(): return currentprocess().getcurrenttask()
def clearcurrenttask(): currentprocess().clearcurrenttask()
#----------------------------------------------------------------------------------------------------



BOX='\u2588'
box='\u2592'
dash='\u2015'
hour=3600
day=24*hour
week=7*day


#def castval(val):
#    for f in [int,cast_str_as_list_(int),float,cast_str_as_list_(float)]:
#        try:
#            return f(val)
#        except:
#            pass
#    return val
#
#
#def cast_str_as_list_(dtype):
#    def cast(s):
#        return [dtype(x) for x in s.split(',')]
#    return cast
#
#
#def parsedef(s):
#    name,val=s.split('=')
#    return name,castval(val)
        

def parse_cmdln_args(cmdargs=sys.argv[1:]):
    cmdargs=deque(cmdargs)
    args=[]
    while len(cmdargs)>0 and '=' not in cmdargs[0]:
        args.append(cmdargs.popleft())

    defs=dict([parsedef(_) for _ in cmdargs])
    return args,defs


def getlossfn():
    return lossfn

def histpath():
    return outpath+'hist'

def logpaths():
    return [outpath+'log','logs/'+util.session.ID]

def getoutpath():
    return outpath

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

def log(msg):
    util.session.log(msg)
    sysutil.write(msg+'\n',*logpaths())
    return act_on_input(checkforinput())


def LOG(msg):
    log('\n\n'+msg+'\n\n')




def providedefault(defs,**kw):
    [(name,defaultval)]=list(kw.items())
    try: return defs[name]
    except: return defaultval


t0=time.perf_counter()
trackedvals=dict()
eventlisteners=dict()

biasinitsize=.1


cmdparams,cmdredefs=parse_cmdln_args()

def getfromargs(**kw):
    return kw[selectone(set(kw.keys()),cmdparams)]

fromcmdparams=getfromargs
getfromcmdparams=getfromargs

plotfineness=50

dash='\u2015'