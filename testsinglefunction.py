from cancellations.display import cdisplay
from cancellations.utilities import tracking, browse, batchjob, sysutil, textutil
from cancellations.examples import profiles as P
import os
import re
import importlib
import sys

#
#fname=sys.argv[1]
#
#pathprofile=browse.defaultpathprofile().butwith(
#    regex='(./)?cancellations.*[a-z].py',\
#    condition1=lambda path: re.search('def '+fname, sysutil.readtextfile(path)),\
#    readinfo=lambda path: sysutil.readtextfile(path))
#
#bprofile=browse.Browse.getdefaultprofile().butwith(
#    msg='select a file to run "{}(*args,**kwargs)" with the args just provided to singlescript.py'.format(fname),
#    parentfolder='cancellations',
#    onlyone=True,
#    options=browse.getpaths(pathprofile)
#)
#
#if __name__=='__main__':
#    path=browse.Browse(**bprofile).run_as_main()
#
#    mname = path.replace('/', '.')[:-3]
#    print('todo: use importlib to run')
#    print(mname)
#
#    #importlib.import_module(mname)
#    m = importlib.import_module(mname)
#    getattr(m, fname)(*sysutil.cmdparams, **sysutil.cmdredefs)
#
#


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
        allpaths=browse.getpaths(pathprofile)
        pattern=re.compile('^def (.*)\(',re.MULTILINE)
        paths,functions=zip(*[(path,f) for path in allpaths for f in pattern.findall(sysutil.readtextfile(pf+path))])
        dotpaths=[path[:-3].replace('/','.') for path in paths]

        options={'{}.{}'.format(path,function):(path,function) for path,function in zip(dotpaths,functions)}

        bprofile=browse.Browse.getdefaultprofile().butwith(\
            onlyone=True,\
            readinfo=lambda path: sysutil.readtextfile(path),\
            options=list(options.keys()),\
            dynamiccondition=lambda path,phrase: re.search(phrase,path)
            )
        bprofile.msg='Press [i] to input filter phrase,\nescape input mode with [ENTER/ESC/arrow keys].\n\n'+\
            'mode: {}\nfilter by: {}\n'+50*textutil.dash+bprofile.msg0
        dotfn=self.runsubprocess(browse.Browse(**bprofile),name='pick function')

        dotpath,selections.fname=options[dotfn]
        selections.dotpath='cancellations.'+dotpath


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
fn(*selections.inputprofile.args,**selections.inputprofile.kwargs)