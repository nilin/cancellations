from cancellations.examples import example
from cancellations.display import cdisplay
from cancellations.utilities import tracking, browse, batchjob, sysutil, config as cfg
import os




batch=tracking.Profile(name='load and train')



batch.task1=browse.Browse
batch.genprofile1=lambda _: browse.getdefaultprofile().butwith(msg='Load target from previous run.',\
    onlyone=True,condition1=(lambda path:os.path.exists(path+'/data/setup') and\
        cfg.agrees(sysutil.parse_metadata(path),n=5,d=2)))


batch.task2=example.Run
batch.genprofile2=lambda prevoutputs: example.getdefaultprofile().butwith(\
    setupdata_path=prevoutputs[0]+'data/setup',n=5)


batch.task3=example.Run
batch.genprofile3=lambda prevoutputs: example.getdefaultprofile().butwith(\
    setupdata_path=prevoutputs[0]+'data/setup',n=5,learnerchoice='ASNN2')


if __name__=='__main__':
    cdisplay.session_in_display(batchjob.Batchjob,batch)
