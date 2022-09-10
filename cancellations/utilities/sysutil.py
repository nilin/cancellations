import os 
import pickle
from ..functions import functions
from . import tracking


def makedirs(filepath):
    path='/'.join(filepath.split('/')[:-1])
    filename=filepath.split('/')[-1]
    os.makedirs(path,exist_ok=True)	

def save(data,*paths,echo=True):
    for path in paths:
        makedirs(path)
        with open(path,'wb') as file:
            pickle.dump(data,file)
    if echo: tracking.log('Saved data to {}'.format(paths))

def savefig(*paths,fig=None):
    for path in paths:
        makedirs(path)
        if fig==None:
            plt.savefig(path)
        else:
            fig.savefig(path)
    tracking.log('Saved figure to {}'.format(paths))


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
    tracking.log('opening path '+path)

    try: subprocess.Popen(['open',path])
    except: pass
    try: subprocess.Popen(['xdg-open',path])
    except: pass
    try: os.startfile(path)
    except: pass

#====================================================================================================

