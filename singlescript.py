from cancellations.examples import example
from cancellations.display import cdisplay
from cancellations.utilities import tracking, browse, batchjob
import os
import importlib




batch=tracking.Profile(name='run a single script')



batch.task1=browse._pickfolders_
batch.genprofile1=lambda _: browse.getdefaultprofile().butwith(\
    parentfolder='cancellations',\
    onlyone=True,\
    #regex='(./)?cancellations.*py',\
    #regex='.*',\
    regex='(./)?cancellations.*[a-z].py',\
    condition1=None,\
    )




if __name__=='__main__':
    [path]=cdisplay.session_in_display(batchjob.runbatch,batch)
    mname=path.replace('/','.')[:-3]
    print('todo: use importlib to run')
    print(mname)
    importlib.import_module(mname)
