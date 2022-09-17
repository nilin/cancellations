from statistics import harmonic_mean
from cancellations.examples import harmonicoscillator1d, estimateobservables
from cancellations.functions import examplefunctions as ef
from cancellations.display import cdisplay
from cancellations.utilities import numutil, tracking, browse, batchjob, energy
import jax
import jax.numpy as jnp
import os


profile=tracking.Profile(tasks=['training','estimate observable'])
#
class Run(batchjob.Batchjob):

    def runbatch(self):

        #task 1

        profile1=harmonicoscillator1d.Run.getdefaultprofile().butwith(iterations=2500)
        psi_descr=self.runsubprocess(harmonicoscillator1d.Run(**profile1),name='training')
        
        
        # task 2

        psi=psi_descr.eval
        E_kin_local=numutil.forfixedparams(energy.genlocalkinetic)(psi)

        profile2=estimateobservables.Run.getdefaultprofile().butwith(\
        p=jax.jit(lambda X: psi(X)**2),\
        qpratio=lambda X: jnp.ones(X.shape[0],),\
    #    observables={'V':lambda X:jnp.sum(X**2,axis=(-2,-1))/2,'K':E_kin_local},\
    #    trueenergies={k:ef.totalenergy(5)/2 for k in ['V','K']},\
        observables={'V':jax.jit(lambda X:jnp.sum(X**2/2,axis=(-2,-1)))},\
        trueenergies={'V':ef.totalenergy(5)/2},\
        maxiterations=10**5)

        self.runsubprocess(estimateobservables.Run(**profile2),name='estimate observable')


if __name__=='__main__':
    Run(**profile).run_as_main()


#
#batch=tracking.Profile(name='harmonic oscillator n=5 d=1')
#
#
#batch.name1='learning'
#batch.task1=harmonicoscillator1d.Run
#batch.genprofile1=lambda _: harmonicoscillator1d.getdefaultprofile().butwith(iterations=2500)
#
#psi0=harmonicoscillator1d.gettarget(estimateobservables.getdefaultprofile()).eval
#p0=lambda X:psi0(X)**2
#
#
#batch.skip2=True
#
#
#
#batch.name3='E[V] of learned psi~, direct method (X~q)'
#batch.task3=estimateobservables.Run
#def genprofile3(prevoutputs):
#    psi_descr=prevoutputs[1]
#    psi=psi_descr.eval
#    E_kin_local=numutil.forfixedparams(energy.genlocalkinetic)(psi)
#
#    profile=estimateobservables.getdefaultprofile().butwith(\
#    p=jax.jit(lambda X: prevoutputs[1].eval(X)**2),\
#    qpratio=lambda X: jnp.ones(X.shape[0],),\
##    observables={'V':lambda X:jnp.sum(X**2,axis=(-2,-1))/2,'K':E_kin_local},\
##    trueenergies={k:ef.totalenergy(5)/2 for k in ['V','K']},\
#    observables={'V':jax.jit(lambda X:jnp.sum(X**2/2,axis=(-2,-1)))},\
#    trueenergies={'V':ef.totalenergy(5)/2},\
#    maxiterations=10**5)
#    return profile
#
#batch.genprofile3=genprofile3
#
#
#
#
#
#
#if __name__=='__main__':
#    #cdisplay.session_in_display(batchjob.Batchjob,batch)
#    #batchjob.Batchjob(batch).run_as_main()
#