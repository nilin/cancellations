from statistics import harmonic_mean
from cancellations.examples import example, harmonicoscillator1d
from cancellations.display import cdisplay
from cancellations.utilities import tracking, browse, batchjob
import os




batch=tracking.Profile(name='harmonic oscillator n=5 d=1')



batch.task1=harmonicoscillator1d.main
batch.genprofile1=lambda prevoutputs: harmonicoscillator1d.getdefaultprofile()



if __name__=='__main__':
    cdisplay.session_in_display(batchjob.runbatch,batch)
