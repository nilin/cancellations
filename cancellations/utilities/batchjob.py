from ..display import cdisplay
from . import tracking

def runbatch(batchprocess):
    batchprofile,display=batchprocess,batchprocess.display

    tasks=[]
    for i in range(1,1000):
        try: tasks.append((batchprofile['task{}'.format(i)],batchprofile['genprofile{}'.format(i)]))
        except: pass

    outputs=[]
    for task, genprofile in tasks:
        outputs.append(cdisplay.runtask(task,genprofile(outputs),display))

    return outputs



class Batchjob(tracking.Process):
    execprocess=runbatch