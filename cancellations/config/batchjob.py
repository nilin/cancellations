from re import I
from cancellations.config import tracking
from cancellations.utilities import textutil
from cancellations.display import _display_
from cancellations.config import browse,tracking




class Batchjob(_display_.Process):
    processname='batchjob'
    processtype='batchjobs'

    def loadprocess(self,process=None,taskname=None):
        assert(tracking.currentprocess()==self)
        if process is None:
            process=tracking.Process()

        process.taskname=taskname
        tracking.loadprocess(process)
        process.display=self.display.blankclone()

        return process,process.display

    @staticmethod
    def unloadprocess():
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

    def run_dummyprocess(self,function,msg=None):
        class Temp(_display_.Process):
            def execprocess(self):
                super().execprocess()
                if msg is not None: tracking.log(msg)
                return function()
        temp=Temp(Temp.getdefaultprofile())
        return self.run_subprocess(temp)

    def pickprofile(self):
        if 'profile' in self: return
        profiles=self.getprofiles()
        profilenamestack=[]
        while not callable(profiles):
            bprofile2=browse.Browse.getdefaultprofile().butwith(\
                onlyone=True,\
                options=list(profiles.keys()),\
                readinfo=lambda profilename:profiles[profilename].__str__()
                )
            bprofile2.msg='select a profile\n'+bprofile2.msg
            profilename=self.run_subprocess(browse.Browse(bprofile2),taskname='pick profile')
            profiles=profiles[profilename]
            profilenamestack.append(profilename)

        genprofile=profiles
        profile=self.run_dummyprocess(genprofile,'generating profile')
        #profile=genprofile()

        profile['profilename']='.'.join(profilenamestack)
        self.profile=profile
        return profile
