from cancellations.config import batchjob, browse, config as cfg, sysutil, tracking
from cancellations.utilities import textutil
import re
import os
import pathlib
import importlib
import sys




class Run(batchjob.Batchjob):

    def runbatch(self):

        fullpaths=[\
            'cancellations/examples/harmonicoscillator2d.py',\
            #'cancellations/examples/harmonicoscillator2d_2.py',\
            'cancellations/examples/Barronnorm.py',\
            'cancellations/plotting/plotmultiple.py']

        bprofile1=browse.Browse.getdefaultprofile().butwith(\
            onlyone=True,\
            readinfo=lambda path: textutil.findblock(sysutil.readtextfile(path),'class Run')[-1],\
            options=fullpaths,\
            displayoption=lambda full:os.path.basename(full)
            )
        bprofile1.msg='select a file to run Run(profile).execprocess().\n\n'\
            +150*textutil.dash\
            +bprofile1.msg


        path=self.run_subprocess(browse.Browse(bprofile1),taskname='pick script')

        path=re.search('([a-z].*)',path).group()
        mname = path.replace('/', '.')[:-3]
        m = importlib.import_module(mname)

        runprofiles=m.Run.getprofiles()
        profilenamestack=[]

        while not isinstance(runprofiles,tracking.Profile):
            bprofile2=browse.Browse.getdefaultprofile().butwith(\
                onlyone=True,\
                options=list(runprofiles.keys()),\
                readinfo=lambda profilename:runprofiles[profilename].__str__()
                )
            bprofile2.msg='select a profile\n'+bprofile2.msg
            profilename=self.run_subprocess(browse.Browse(bprofile2),taskname='pick profile')
            runprofiles=runprofiles[profilename]
            profilenamestack.append(profilename)

        runprofile=runprofiles
        runprofile['profilename']='/'.join(profilenamestack)

        self.run=m.Run(runprofile)

        # task 3

        if cfg.debug:
            cfg.postprocesses.append(self.run)
            cfg.display_on=False
            return

        self.run_subprocess(self.run,taskname='run script')


    @staticmethod
    def getdefaultprofile(**kw):
        return batchjob.Batchjob.getdefaultprofile().butwith(tasks=['pick script','pick profile','run script'],**kw)


def main():
    Run().run_as_main()
    cfg.run_afterdisplayclosed()

def debug():
    from cancellations.config import sysutil
    import jax
    sysutil.clearscreen()
    cfg.debug=True
    with jax.disable_jit():
        main()

if __name__=='__main__':
    if 'd' in sys.argv:
        debug()
    else:
        main()