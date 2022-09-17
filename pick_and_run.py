from cancellations.display import cdisplay
from cancellations.utilities import tracking, browse, batchjob, sysutil
import os
import re
import importlib


batch = tracking.Profile(name='run a single script')

batch.name1='browse'
batch.task1 = browse.Browse
batch.genprofile1 = lambda _: browse.getdefaultprofile().butwith(
    msg='select a file to run Run.execprocess(self)',
    parentfolder='cancellations',
    onlyone=True,
    regex='(./)?cancellations/examples.*[a-z].py',
    condition1=lambda path: re.search('class Run', sysutil.readtextfile(path)),
    readinfo=lambda path: sysutil.readtextfile(path)
)


def execprocess(process):
    mname = process.path.replace('/', '.')[:-3]
    m = importlib.import_module(mname)
    run=m.Run(m.getdefaultprofile(),process.display)
    tracking.loadprocess(run)
    run.execprocess()
    tracking.unloadprocess(run)

batch.name2='run script'
batch.task2=tracking.newprocess(execprocess)
batch.genprofile2=lambda prevoutputs: tracking.Profile(path=prevoutputs[1])

cdisplay.session_in_display(batchjob.Batchjob, batch)

