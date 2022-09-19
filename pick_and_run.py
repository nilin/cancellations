from cancellations.display import cdisplay
from cancellations.utilities import tracking, browse, batchjob, sysutil, textutil
from cancellations.examples import profiles as P
import os
import re
import importlib



profile=tracking.Profile(tasks=['pick script','pick profile','run script'])

class Run(batchjob.Batchjob):

    def runbatch(self):

        # task 1
        pf='cancellations/examples/'
        pathprofile=browse.defaultpathprofile().butwith(\
            parentfolder=pf,\
            regex='.*[a-z].py',\
            condition1=lambda path: re.search('class Run', sysutil.readtextfile(pf+path)),\
            )

        relpaths=browse.getpaths(pathprofile)
        relpaths=sorted(relpaths,key=lambda relpath: 1000 if 'unsupervised' in relpath else 1)
        fullpaths=[pf+relpath for relpath in relpaths]
        rels={full:rel for full,rel in zip(fullpaths,relpaths)}

        bprofile=browse.Browse.getdefaultprofile().butwith(\
            onlyone=True,\
            readinfo=lambda path: textutil.startingfrom(sysutil.readtextfile(path),'class Run'),\
            options=fullpaths,\
            displayoption=lambda full:rels[full]
            )
        bprofile.msg='select a file to run Run(profile).execprocess().\n\n'\
            +'Press [b] to run a single function instead.\n'+50*textutil.dash\
            +bprofile.msg0
        path=self.runsubprocess(browse.Browse(**bprofile),name='pick script')


        # postprocess

        path=re.search('([a-z].*)',path).group()
        mname = path.replace('/', '.')[:-3]
        m = importlib.import_module(mname)


        # task 2

        profilegenerators=P.getprofiles(m.Run.exname)
        bprofile=browse.Browse.getdefaultprofile().butwith(\
            onlyone=True,\
            #readinfo=lambda pname: textutil.startingfrom(sysutil.readtextfile('cancellations/examples/profiles.py'),m.Run.exname,pname),\
            options=list(profilegenerators.keys())
            )
        bprofile.msg='select a profile\n'+bprofile.msg
        profilename=self.runsubprocess(browse.Browse(**bprofile),name='pick profile')
        profile=profilegenerators[profilename]()
        profile['profilename']=profilename


        # task 3

        self.runsubprocess(m.Run(**profile),name='run script')




if __name__=='__main__':
    Run(**profile).run_as_main()

