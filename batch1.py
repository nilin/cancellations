from cancellations.examples import example
from cancellations.display import cdisplay
from cancellations.utilities import util, browse, batchjob
import os




batch=util.Profile(name='train 2 times')




batch.task1=example.main
batch.genprofile1=lambda prevoutputs: example.getdefaultprofile()


batch.task2=example.main
batch.genprofile2=lambda prevoutputs: example.getdefaultprofile().butwith(learnerchoice='ASNN2')


if __name__=='__main__':
    cdisplay.session_in_display(batchjob.runbatch,batch)