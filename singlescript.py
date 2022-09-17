from cancellations.display import cdisplay
from cancellations.utilities import tracking, browse, batchjob, sysutil
import os
import re
import importlib


fname='test'

batch = tracking.Profile(name='run a single script')

batch.name1='browse'
batch.task1 = browse.Browse
batch.genprofile1 = lambda _: browse.getdefaultprofile().butwith(
    msg='select a file to run "{}(*args,**kwargs)" with the args just provided to singlescript.py'.format(fname),
    parentfolder='cancellations',
    onlyone=True,
    regex='(./)?cancellations.*[a-z].py',
    condition1=lambda path: re.search('def run', sysutil.readtextfile(path)),
    readinfo=lambda path: sysutil.readtextfile(path)
)




if __name__=='__main__':
    [path] = cdisplay.session_in_display(batchjob.Batchjob, batch)
    mname = path.replace('/', '.')[:-3]
    print('todo: use importlib to run')
    print(mname)

    #importlib.import_module(mname)
    m = importlib.import_module(mname)
    getattr(m, fname)(*sysutil.cmdparams, **sysutil.cmdredefs)
