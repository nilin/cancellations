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
        raise NotImplementedError


    def loadprocess(self,process=None,taskname=None):
        assert(tracking.currentprocess()==self)
        if process is None:
            process=tracking.Process()

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
