from statistics import harmonic_mean
from cancellations.examples import example, harmonicoscillator1d, estimateobservables
from cancellations.functions import examplefunctions as ef
from cancellations.display import cdisplay
from cancellations.utilities import tracking, arrayutil, browse, batchjob, energy, sysutil
import jax
import jax.numpy as jnp
import os




batch=tracking.Profile(name='harmonic oscillator n=5 d=1')


batch.name1='pick run'
batch.task1=browse.Browse
batch.genprofile1=lambda _: browse.getdefaultprofile().butwith(onlyone=True)




batch.name2='extract learner'
batch.task2=tracking.getprocess(lambda process: sysutil.load(process.path).learner.restore())
batch.genprofile2=lambda prevoutputs: tracking.Profile(path=prevoutputs[0]+'data/unprocessed')





batch.name3='E[V] of learned psi~, direct method (X~q)'
batch.task3=estimateobservables.Run
batch.genprofile3=lambda prevoutputs: estimateobservables.getdefaultprofile().butwith(\
    p=jax.jit(lambda X: prevoutputs[-1].eval(X)**2),\
    qpratio=lambda X: jnp.ones(X.shape[0],),\
    trueenergies=[ef.totalenergy(5)/2])



if __name__=='__main__':
    cdisplay.session_in_display(batchjob.Batchjob,batch)
