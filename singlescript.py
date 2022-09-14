from cancellations.examples import example
from cancellations.display import cdisplay
from cancellations.utilities import tracking, browse, batchjob, sysutil
import os
import re
import importlib


batch = tracking.Profile(name='run a single script')

batch.name1='browse'
batch.task1 = browse.Browse
batch.genprofile1 = lambda _: browse.getdefaultprofile().butwith(
    msg='select a file to run test(*args,**kwargs) with the args just provided to singlescript.py',
    parentfolder='cancellations',
    onlyone=True,
    regex='(./)?cancellations.*[a-z].py',
    condition1=lambda path: re.search('def test', sysutil.readtextfile(path)),
    readinfo=lambda path: sysutil.readtextfile(path)
)


if __name__ == '__main__':
    [path] = cdisplay.session_in_display(batchjob.Batchjob, batch)
    mname = path.replace('/', '.')[:-3]
    print('todo: use importlib to run')
    print(mname)
    m = importlib.import_module(mname)
    getattr(m, 'test')(*sysutil.cmdparams, **sysutil.cmdredefs)
