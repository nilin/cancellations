from cancellations.display import cdisplay
from cancellations.utilities import tracking, browse, batchjob, sysutil, textutil
from cancellations.examples import profiles as P
import os
import re
import importlib
import sys


selections=tracking.dotdict()

profile=tracking.Profile(tasks=['pick function','pick inputs'])

class Run(batchjob.Batchjob):

    def runbatch(self):

        # task 1

        pf='cancellations/'
        pathprofile=browse.defaultpathprofile().butwith(\
            parentfolder=pf,\
            regex='.*[a-z].py',\
            condition1=None
            #condition1=lambda path: re.search('class Run', sysutil.readtextfile(path)),\
            )
        allrelpaths=browse.getpaths(pathprofile)
        allfullpaths=[pf+relpath for relpath in browse.getpaths(pathprofile)]
        rels={full:rel for full,rel in zip(allfullpaths,allrelpaths)}

        pattern=re.compile('^def (.*)\(',re.MULTILINE)

        fullfnpaths=[(fullpath,f) for fullpath in allfullpaths for f in pattern.findall(sysutil.readtextfile(fullpath))]
#        reldotpaths=[(fullpath.replace('/','.'),f) for fullpath in allfullpaths for f in pattern.findall(sysutil.readtextfile(fullpath))]
#        fulldotpaths=[full[:-3].replace('/','.') for full in fullpaths]
#        reldotpaths=[rels[full][:-3].replace('/','.') for full in fullpaths]
#
#        options={'{}.{}'.format(dotpath,function):(dotpath,function) for dotpath,function in zip(dotpaths,functions)}
#        slashoptions={'{}.{}'.format(dotpath,function):(slashpath,function) for dotpath,function,slashpath in zip(dotpaths,functions,paths)}
#
        def readinfo(pair):
            full,fn=pair
            return textutil.startingfrom(sysutil.readtextfile(full),'def '+fn)

        def displayoption(pair):
            full,fn=pair
            out=rels[full].replace('/','.')[:-3]+'.'+fn
            return out

        bprofile=browse.Browse.getdefaultprofile().butwith(\
            onlyone=True,\
            options=fullfnpaths,\
            displayoption=displayoption,\
            dynamiccondition=lambda fulldotpath,phrase: re.search(phrase,fulldotpath),\
            readinfo=readinfo\
            )
        bprofile.msg='Press [i] to input filter phrase,\nescape input mode with arrow keys.\n\n'+\
            'mode: {}\nfilter by: {}\n'+50*textutil.dash+bprofile.msg0
        fullpath,fname=self.runsubprocess(browse.Browse(**bprofile),name='pick function')

        selections.dotpath,selections.fname=fullpath.replace('/','.')[:-3],fname

        # task 2

        profilegenerators=P.get_test_fn_inputs()
        bprofile=browse.Browse.getdefaultprofile().butwith(\
            onlyone=True,\
            readinfo=lambda : sysutil.readtextfile('cancellations/examples/profiles.py'),\
            options=list(profilegenerators.keys())
            )
        bprofile.msg='select an input profile\n'+bprofile.msg
        profilename=self.runsubprocess(browse.Browse(**bprofile),name='pick inputs')

        selections.inputprofile=profilegenerators[profilename]()



Run(**profile).run_as_main()
sysutil.clearscreen()

m = importlib.import_module(selections.dotpath)
fn=getattr(m,selections.fname)
out=fn(*selections.inputprofile.args,**selections.inputprofile.kwargs)
print(out)