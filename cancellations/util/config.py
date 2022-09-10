
processes=[]

session=Process({'name':'session'},display=None,ID='session '+nowstr())



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


def getlossfn():
    return lossfn


def histpath():
    return outpath+'hist'

def logpaths():
    return [outpath+'log','logs/'+session.ID]

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

def formatvars(elements,separator=' ',ignore={}):
    return separator.join(['{}={}'.format(name,val) for name,val in elements if name not in ignore])


#----------------------------------------------------------------------------------------------------

def log(msg):
    session.log(msg)
    return act_on_input(checkforinput())


def LOG(msg):
    log('\n\n'+msg+'\n\n')



def indent(s):
    return '\n'.join(['    '+l for l in s.splitlines()])



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