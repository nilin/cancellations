from statistics import harmonic_mean
from cancellations.examples import example, harmonicoscillator1d, estimateobservables
from cancellations.display import cdisplay
from cancellations.utilities import tracking, browse, batchjob
import os




batch=tracking.Profile(name='harmonic oscillator n=5 d=1')



batch.task1=harmonicoscillator1d.Run
batch.genprofile1=lambda _: harmonicoscillator1d.getdefaultprofile()


batch.task2=estimateobservables.Run
batch.genprofile2=lambda prevoutputs: estimateobservables.getdefaultprofile().butwith(\
    wavefunction=prevoutputs[0].eval)


if __name__=='__main__':
    cdisplay.session_in_display(batchjob.Batchjob,batch)
