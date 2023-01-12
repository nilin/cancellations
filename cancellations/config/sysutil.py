import os 
import pickle
from collections import deque
import matplotlib.pyplot as plt
import sys
from cancellations.config import config as cfg, tracking
from cancellations.utilities import textutil


def makedirs(filepath):
    path='/'.join(filepath.split('/')[:-1])
    filename=filepath.split('/')[-1]
    os.makedirs(path,exist_ok=True)    

def save(data,path,echo=True):
    makedirs(path)
    with open(path,'wb') as file:
        pickle.dump(data,file)
    if echo: tracking.log('Saved data to {}'.format(path))

def savefig(path,fig=None):
    makedirs(path)
    if fig is None: plt.savefig(path)
    else: fig.savefig(path)
    tracking.log('Saved figure to {}'.format(path))


def write(msg,path,mode='a'):
    makedirs(path)
    with open(path,mode) as f:
        f.write(msg)
    
def load(path):
    with open(path,"rb") as file:
        return pickle.load(file)

def read(path):
    with open(path,'r') as f:
        return f.readlines()
        
def showfile(path):
    import os
    import subprocess
    cfg.session.log('opening path '+path)

    try: subprocess.Popen(['open',path])
    except: pass
    try: subprocess.Popen(['xdg-open',path])
    except: pass
    try: os.startfile(path)
    except: pass

def removetemplog(path):
    delpaths=[os.path.join(path,fn) for fn in os.listdir(path) if '.templog' in fn]
    for delpath in delpaths:
        assert(readtextfile(delpath)=='temporary log')
        os.remove(delpath)


def filebrowserlog(path,msg):
    removetemplog(path)
    write('temporary log',os.path.join(path,'___log_'+textutil.cleanstring(msg)[:25]+'___.templog'),mode='w')
    cfg.postcommands.append(lambda: removetemplog(path))


#====================================================================================================
#====================================================================================================


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
        

def parse_args(cmdargs=sys.argv[1:]):
    cmdargs=deque(cmdargs)
    args=[]
    while len(cmdargs)>0 and '=' not in cmdargs[0]:
        args.append(cmdargs.popleft())

    defs=dict([parsedef(_) for _ in cmdargs])
    return args,defs

def parse_metadata(path):
    with open(path+'metadata.txt','r') as f:
        return parse_args(f.readline())[1]

def readtextfile(path):
    with open(path,'r') as f:
        return ''.join(f.readlines())


cmdparams,cmdredefs=parse_args()

def commonanc(*fs):
    levels=list(zip(*[f.split('/') for f in fs]))
    
    path=''
    difflevel=[]
    for l in levels:
        if all([li==l[0] for li in l]):
            path+=l[0]+'/'
        else:
            break
    return path,[f[len(path):] for f in fs]


def clearscreen():
    os.system('cls' if os.name == 'nt' else 'clear')


def maybe(f,ifnot=None):
    def F(*args,**kwargs):
        try:
            return f(*args,**kwargs)
        except:
            return ifnot
    return F


def test():
    print(readtextfile('batch1.py'))


def tryexcept(f,g,xs,ys):
    try:
        return f(*xs)
    except:
        return g(*ys)


write('setup','outputs/setup.txt')