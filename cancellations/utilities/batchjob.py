from re import I
from cancellations.utilities import textutil, config as cfg
from ..display import _display_
from . import tracking






class Batchjob(_display_.Process):
    processname='batchjob'

    def execprocess(self):
        self.dashboard=self.display
        self.tasklistdisplay,self.subdisplay=self.dashboard.vsplit(limits=[3])
        self.tasklisttext=self.tasklistdisplay.add(0,0,_display_._TextDisplay_(''))
        #self.tasklistdisplay.outline=True
        #self.tasklistdisplay.arm()
        #self.tasklistdisplay.draw()
        self.runbatch()


    def runbatch(self):
        process,display=self.loadprocess()
        process,display=self.swap_process()


    def loadprocess(self,process=None,taskname=None):
        assert(tracking.currentprocess()==self)
        if process==None:
            process=tracking.Process()
        #elif isinstance(process,tracking.Profile):
        #    process=tracking.Process(process)

        process.taskname=taskname
        self.printtaskline(process)

        tracking.loadprocess(process)
        process.display=self.subdisplay.blankclone()

        return process,process.display

    def unloadprocess(self):
        _display_.clearcurrentdash()
        tracking.unloadprocess()

    def swap_process(self,process=None):
        self.unloadprocess()
        return self.loadprocess(process)

    def run_subprocess(self,subprocess: _display_.Process,**kw):
        self.loadprocess(subprocess,**kw)
        out=subprocess.execprocess()
        self.unloadprocess()
        return out

    def printtaskline(self,process):
        self.tasklisttext.msg='    '.join(['*{}*'.format(task) if task==process.taskname else task for task in self.profile.tasks])+\
            '\n'+self.dashboard.width*textutil.dash
        self.tasklistdisplay.arm()
        self.tasklistdisplay.draw()

    @staticmethod
    def getdefaultprofile(**kw):
        return tracking.Process.getdefaultprofile().butwith(tasks=[],**kw)

#        self.task=name
#        self.headlinedisplay().msg=\
#            '    '.join(['>{}<'.format(task) if task==self.getval('task') else task for task in self.profile.tasks])+\
#            '\n'+self.display.width*textutil.dash
#        _display_.getscreen().getch(); self.tasklistcdisplay.draw(); _display_.getscreen().refresh()

    #    return subprocess.run_in_display(*self.subdisplay)

#    def execprocess(batchprocess):
#        batchprofile,dashboard=batchprocess,batchprocess.display
#        batchprocess.prepdisplay()
#        tasks=[]
#        for i in range(1,1000):
#            if 'skip{}'.format(i) in batchprofile.keys(): continue
#            try: tasks.append((batchprofile['name{}'.format(i)],batchprofile['task{}'.format(i)],batchprofile['genprofile{}'.format(i)]))
#            except: break
#
#        tasknames=[name for name,_,_ in tasks]
#        outputs=[None]
#        for i, (name, task, genprofile) in enumerate(tasks):
#            batchprocess.headlinedisplay().msg='tasks:        '+'        '.join(tasknames[:i]+['> '+name+' <']+tasknames[i+1:])+\
#            '\n'+dashboard.width*textutil.dash #+'current task: '+task.ID
#            cfg.screen.getch(); batchprocess.tasklistcdisplay.draw(); cfg.screen.refresh()
#
#            outputs.append(cdisplay.runtask(task,genprofile(outputs).butwith(taskname=name),batchprocess.subdisplay))
#
#        return outputs


        #self.tasklistcdisplay,_=dashboard.add(cdisplay.ConcreteStackedDisplay(dashboard.xlim,(dashboard.ylim[0],dashboard.ylim[0]+2)))
        #self.tasklistcdisplay.add(disp.StaticText(msg=''),name='textdisplay')
        #self.subdisplay,_=dashboard.add(_display_.Dashboard(\
        #    (dashboard.xlim[0]+3,dashboard.xlim[1]-3),(dashboard.ylim[0]+4,dashboard.ylim[1]-2)))

    #def headlinedisplay(self):
    #    return self.tasklistcdisplay.elements['textdisplay']
