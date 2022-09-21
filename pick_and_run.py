from cancellations.utilities import setup
from cancellations.display import _display_
from cancellations.utilities import tracking, browse, batchjob, sysutil, textutil
from cancellations.examples import profiles as P
import os
import re
import importlib




class Run(batchjob.Batchjob):
    processname='pick_and_run'

    def runbatch(self):



        #self.loadprocess()



        pf='cancellations/examples/'
        pathprofile=browse.defaultpathprofile().butwith(\
            parentfolder=pf,\
            regex='.*[a-z].py',\
            condition=lambda path: re.search('class Run', sysutil.readtextfile(pf+path)),\
            )

        relpaths=browse.getpaths(pathprofile)
        relpaths=sorted(relpaths,key=lambda relpath: 1000 if 'unsupervised' in relpath else 1)
        fullpaths=[pf+relpath for relpath in relpaths]
        rels={full:rel for full,rel in zip(fullpaths,relpaths)}

        bprofile1=browse.Browse.getdefaultprofile().butwith(\
            onlyone=True,\
            readinfo=lambda path: textutil.startingfrom(sysutil.readtextfile(path),'class Run'),\
            options=fullpaths,\
            displayoption=lambda full:rels[full]
            )
        bprofile1.msg='select a file to run Run(profile).execprocess().\n\n'\
            +'Press [b] to run a single function instead.\n'+50*textutil.dash\
            +bprofile1.msg0



        path=self.run_subprocess(browse.Browse(bprofile1))

#        browsingprocess1,display=self.swap_process(bprofile1)
#        path=browse.browse(browsingprocess1)



        path=re.search('([a-z].*)',path).group()
        mname = path.replace('/', '.')[:-3]
        m = importlib.import_module(mname)
        runprofiles=P.getprofiles(m.Run.processname)
        bprofile2=browse.Browse.getdefaultprofile().butwith(\
            onlyone=True,\
            #readinfo=lambda pname: textutil.startingfrom(sysutil.readtextfile('cancellations/examples/profiles.py'),m.Run.exname,pname),\
            options=list(runprofiles.keys()),\
            readinfo=lambda profilename:runprofiles[profilename].__str__()
            )
        bprofile2.msg='select a profile\n'+bprofile2.msg




        profilename=self.run_subprocess(browse.Browse(bprofile2),name='pick profile')

#        browsingprocess2,display=self.swap_process(bprofile2)
#        profilename=browse.browse(browsingprocess2)




        runprofile=runprofiles[profilename]
        runprofile['profilename']=profilename


        # task 3

        self.run_subprocess(m.Run(runprofile),name='run script')



profile=tracking.Profile(tasks=['pick script','pick profile','run script'])

if __name__=='__main__':
    Run(profile).run_as_main()

