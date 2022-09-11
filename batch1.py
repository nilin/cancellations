from cancellations.examples import example
from cancellations.display import cdisplay
from cancellations.utilities import tracking, browse, batchjob
import os




batch=tracking.Profile(name='train 2 times')




batch.task1=example.main
batch.genprofile1=lambda prevoutputs: example.getdefaultprofile().\
    butwith(n=5,adjusttargetsamples=1000,samples_train=10000,iterations=1000)


batch.task2=example.main
batch.genprofile2=lambda prevoutputs: example.getdefaultprofile().butwith(learnerchoice='ASNN2',n=5)


if __name__=='__main__':
    cdisplay.session_in_display(batchjob.runbatch,batch)
