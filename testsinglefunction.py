from cancellations.display import cdisplay
from cancellations.utilities import tracking, browse, batchjob, sysutil
import os
import re
import importlib
import sys


fname=sys.argv[1]

pathprofile=browse.defaultpathprofile().butwith(
    regex='(./)?cancellations.*[a-z].py',\
    condition1=lambda path: re.search('def '+fname, sysutil.readtextfile(path)),\
    readinfo=lambda path: sysutil.readtextfile(path))

bprofile=browse.Browse.getdefaultprofile().butwith(
    msg='select a file to run "{}(*args,**kwargs)" with the args just provided to singlescript.py'.format(fname),
    parentfolder='cancellations',
    onlyone=True,
    options=browse.getpaths(pathprofile)
)

if __name__=='__main__':
    path=browse.Browse(**bprofile).run_as_main()

    mname = path.replace('/', '.')[:-3]
    print('todo: use importlib to run')
    print(mname)

    #importlib.import_module(mname)
    m = importlib.import_module(mname)
    getattr(m, fname)(*sysutil.cmdparams, **sysutil.cmdredefs)

