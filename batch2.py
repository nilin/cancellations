from cancellations.examples import example
from cancellations.display import cdisplay
from cancellations.utilities import tracking, browse, batchjob
import os




batch=tracking.Profile(name='load and train')



batch.task1=browse._pickfolders_
batch.genprofile1=lambda _: browse.getdefaultprofile().butwith(\
    onlyone=True,condition1=(lambda path:os.path.exists(path+'/data/setup')))



batch.task2=example.main
batch.genprofile2=lambda prevoutputs: example.getdefaultprofile().butwith(\
    setupdata_path=prevoutputs[0]+'data/setup',n=4)




if __name__=='__main__':
    cdisplay.session_in_display(batchjob.runbatch,batch)
