from cancellations.display import cdisplay
from cancellations.utilities import tracking, browse, batchjob, sysutil
import os
import re
import importlib



profile=tracking.Profile(tasks=['pick script','run script'])

class Run(batchjob.Batchjob):

    def runbatch(self):

        # task 1

        bprofile=browse.getdefaultprofile().butwith(
            msg='select a file to run Run.execprocess(self)',
            #parentfolder='cancellations',
            parentfolder='.',
            onlyone=True,
            #regex='(./)?cancellations.*[a-z].py',
            #regex='.*[a-z].py',
            regex='(.?/?[^/]*|.*cancellations/examples/.*)[a-z].py',
            condition1=lambda path: re.search('class Run', sysutil.readtextfile(path)),
            readinfo=lambda path: sysutil.readtextfile(path)
        )
        path=self.runsubprocess(browse.Browse(**bprofile),name='pick script')


        # task 2

        path=re.search('([a-z].*)',path).group()
        mname = path.replace('/', '.')[:-3]
        m = importlib.import_module(mname)
        self.runsubprocess(m.Run(**m.profile),name='run script')

#
#if __name__=='__main__':
#    Run(**profile).run_as_main()
#
