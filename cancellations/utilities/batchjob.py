from cancellations.utilities import textutil, config as cfg
from ..display import cdisplay,display as disp
from . import tracking

def runbatch(batchprocess):
    batchprofile,dashboard=batchprocess,batchprocess.display
    batchprocess.prepdisplay()
    tasks=[]
    for i in range(1,1000):
        if 'skip{}'.format(i) in batchprofile.keys(): continue
        try: tasks.append((batchprofile['name{}'.format(i)],batchprofile['task{}'.format(i)],batchprofile['genprofile{}'.format(i)]))
        except: break

    tasknames=[name for name,_,_ in tasks]
    outputs=[None]
    for i, (name, task, genprofile) in enumerate(tasks):
        batchprocess.headlinedisplay().msg='tasks:        '+'        '.join(tasknames[:i]+['> '+name+' <']+tasknames[i+1:])+\
        '\n'+dashboard.width*textutil.dash #+'current task: '+task.ID
        cfg.screen.getch(); batchprocess.tasklistcdisplay.draw(); cfg.screen.refresh()

        outputs.append(cdisplay.runtask(task,genprofile(outputs).butwith(taskname=name),batchprocess.subdisplay))

    return outputs





class Batchjob(tracking.Process):
    execprocess=runbatch

    def prepdisplay(self):
        dashboard=self.display
        self.tasklistcdisplay,_=dashboard.add(cdisplay.ConcreteDisplay(dashboard.xlim,(0,2)))
        self.tasklistcdisplay.add(disp.StaticText(msg=''),name='textdisplay')
        self.subdisplay,_=dashboard.add(cdisplay.Dashboard((3,dashboard.xlim[1]-3),(2+2,dashboard.ylim[1]-2)))

    def headlinedisplay(self):
        return self.tasklistcdisplay.elements['textdisplay']
