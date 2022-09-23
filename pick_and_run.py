from cancellations.utilities import setup
from cancellations.display import _display_
from cancellations.utilities import tracking, browse, batchjob, sysutil, textutil
from cancellations.examples import profiles as P
import os
import re
import importlib
import sys




class Run(batchjob.Batchjob):
    #processname='pick_and_run'

    def runbatch(self):



        #self.loadprocess()



        pf='cancellations/'#examples/'
        pathprofile=browse.defaultpathprofile().butwith(\
            parentfolder=pf,\
            regex='.*[a-z].py',\
            condition=lambda path: \
                re.search('class Run[^a-z]', sysutil.readtextfile(pf+path))\
                and not re.search('ignore pick_and_run', sysutil.readtextfile(pf+path))\
                and not re.search('dontpick', sysutil.readtextfile(pf+path)),\
            )

        relpaths=browse.getpaths(pathprofile)
        relpaths=sorted(relpaths,key=lambda relpath: 1000 if 'unsupervised' in relpath else 1)
        fullpaths=[pf+relpath for relpath in relpaths]
        rels={full:rel for full,rel in zip(fullpaths,relpaths)}

        bprofile1=browse.Browse.getdefaultprofile().butwith(\
            onlyone=True,\
            #readinfo=lambda path: textutil.startingfrom(sysutil.readtextfile(path),'class Run',linewise=True),\
            readinfo=lambda path: textutil.findblock(sysutil.readtextfile(path),'class Run')[-1],\
            options=fullpaths,\
            displayoption=lambda full:rels[full]
            )
        bprofile1.msg='select a file to run Run(profile).execprocess().\n\n'\
            +150*textutil.dash\
            +bprofile1.msg



        path=self.run_subprocess(browse.Browse(bprofile1),taskname='pick script')

#        browsingprocess1,display=self.swap_process(bprofile1)
#        path=browse.browse(browsingprocess1)



        path=re.search('([a-z].*)',path).group()
        mname = path.replace('/', '.')[:-3]
        m = importlib.import_module(mname)
        processname=mname.split('.')[-1]
        runprofiles=P.getprofiles(processname)
        bprofile2=browse.Browse.getdefaultprofile().butwith(\
            onlyone=True,\
            #readinfo=lambda pname: textutil.startingfrom(sysutil.readtextfile('cancellations/examples/profiles.py'),m.Run.exname,pname),\
            options=list(runprofiles.keys()),\
            readinfo=lambda profilename:runprofiles[profilename].__str__()
            )
        bprofile2.msg='select a profile\n'+bprofile2.msg




        profilename=self.run_subprocess(browse.Browse(bprofile2),taskname='pick profile')

#        browsingprocess2,display=self.swap_process(bprofile2)
#        profilename=browse.browse(browsingprocess2)



        if profilename==None:
            runprofile=m.Run.getdefaultprofile()
        else:
            runprofile=runprofiles[profilename]
            runprofile['profilename']=profilename

        self.run=m.Run(runprofile)

        # task 3

        if setup.debug:
            setup.postrun=self.run
            setup.display_on=False
            return

        self.run_subprocess(self.run,taskname='run script')


    @staticmethod
    def getdefaultprofile(**kw):
        return batchjob.Batchjob.getdefaultprofile().butwith(tasks=['pick script','pick profile','run script'],**kw)


def main():
    if 'debug' in sys.argv: setup.debug=True

    Run().run_as_main()
    setup.postrun.run_as_NODISPLAY()


if __name__=='__main__': main()