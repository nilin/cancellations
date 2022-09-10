import example
import display.cdisplay as cdisplay
from display.cdisplay import runtask
import config as cfg
import config.browse as browse
import os

batch=cfg.Profile(name='load and train')





batch.task1=browse._pickfolders_
batch.genprofile1=lambda _: browse.getdefaultprofile().butwith(\
    onlyone=True,condition=(lambda path:os.path.exists(path+'/data/setup')))



batch.task2=example.main
batch.genprofile2=lambda prevoutputs: example.getdefaultprofile().butwith(\
    setupdata_path=prevoutputs[0]+'data/setup')




def runbatch(batch,display):
    tasks=[]
    for i in range(1,1000):
        try: tasks.append((batch['task{}'.format(i)],batch['genprofile{}'.format(i)]))
        except: pass

    outputs=[]
    for task, genprofile in tasks:
        outputs.append(runtask(task,genprofile(outputs),display))


if __name__=='__main__':
    cdisplay.session_in_display(runbatch,batch)