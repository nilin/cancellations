from statistics import harmonic_mean
from cancellations.examples import example, harmonicoscillator1d,estimateobservables
from cancellations.functions import examplefunctions as ef
from cancellations.display import cdisplay
from cancellations.utilities import tracking, browse, batchjob
import os




batch=tracking.Profile(name='harmonic oscillator n=5 d=1')


batch.task1=estimateobservables.Run
batch.genprofile1=lambda _: estimateobservables.getdefaultprofile().butwith(\
    wavefunction=harmonicoscillator1d.gettarget(estimateobservables.getdefaultprofile()).eval,
    trueenergy=ef.totalenergy(5)/2)
#batch.task2=harmonicoscillator1d.Run
#batch.genprofile2=lambda prevoutputs: harmonicoscillator1d.getdefaultprofile()



if __name__=='__main__':
    cdisplay.session_in_display(batchjob.Batchjob,batch)
