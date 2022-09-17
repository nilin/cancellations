from cancellations.display import cdisplay
from cancellations.utilities import tracking, browse, batchjob, sysutil
from cancellations.examples import profiles as P
import os
import re
import importlib



profile=tracking.Profile(tasks=['pick script','run script'])

class Run(batchjob.Batchjob):

    def runbatch(self):

        # task 1

        pathprofile=browse.defaultpathprofile().butwith(\
            parentfolder='.',\
            #regex='(.?/?[^/]*|.*cancellations/examples/.*)[a-z].py',\
            regex='(.*cancellations/examples/.*)[a-z].py',\
            condition1=lambda path: re.search('class Run', sysutil.readtextfile(path)),\
            )
        bprofile=browse.Browse.getdefaultprofile().butwith(\
            msg='select a file to run Run.execprocess(self)',\
            onlyone=True,\
            readinfo=lambda path: sysutil.readtextfile(path),\
            options=browse.getpaths(pathprofile)
            )
        path=self.runsubprocess(browse.Browse(**bprofile),name='pick script')

        # postprocess

        path=re.search('([a-z].*)',path).group()
        mname = path.replace('/', '.')[:-3]
        m = importlib.import_module(mname)


        # task 2

        profilegenerators=P.getprofiles(m.Run.exname)
        bprofile=browse.Browse.getdefaultprofile().butwith(\
            msg='select a profile',\
            onlyone=True,\
            readinfo=lambda : sysutil.readtextfile('cancellations/examples/profiles.py'),\
            options=list(profilegenerators.keys())
            )
        profilename=self.runsubprocess(browse.Browse(**bprofile),name='pick profile')
        profile=profilegenerators[profilename]()



        # task 3

        self.runsubprocess(m.Run(**profile),name='run script')


if __name__=='__main__':
    Run(**profile).run_as_main()

