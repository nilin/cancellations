from statistics import harmonic_mean
from cancellations.examples import example, harmonicoscillator1d, estimateobservables
from cancellations.functions import examplefunctions as ef
from cancellations.display import cdisplay
from cancellations.utilities import tracking, arrayutil, browse, batchjob, energy
import jax
import jax.numpy as jnp
import os




batch=tracking.Profile(name='harmonic oscillator n=5 d=1')


batch.name1='learning'
batch.task1=harmonicoscillator1d.Run
batch.genprofile1=lambda _: harmonicoscillator1d.getdefaultprofile().butwith(iterations=2500)

psi0=harmonicoscillator1d.gettarget(estimateobservables.getdefaultprofile()).eval
p0=lambda X:psi0(X)**2


batch.skip2=True

#batch.name2='E[V] of learned psi~, indirect method (X~p0)'
#batch.task2=estimateobservables.Run
#batch.genprofile2=lambda prevoutputs: estimateobservables.getdefaultprofile().butwith(\
#    p=p0,\
#    qpratio=jax.jit(lambda X: prevoutputs[0].eval(X)**2/p0(X)),\
#    trueenergies=[ef.totalenergy(5)/2])


batch.name3='E[V] of learned psi~, direct method (X~q)'
batch.task3=estimateobservables.Run
batch.genprofile3=lambda prevoutputs: estimateobservables.getdefaultprofile().butwith(\
    p=jax.jit(lambda X: prevoutputs[0].eval(X)**2),\
    qpratio=lambda X: jnp.ones(X.shape[0],),\
    trueenergies=[ef.totalenergy(5)/2])


batch.skip4=True

#batch.name4='Total energy K+V for learned state'
#batch.task4=estimateobservables.Run
#def genprofile4(prevoutputs):
#    psi=prevoutputs[0].f
#    E_kin_local=energy.genlocalkinetic(psi)
#    total_localenergy=jax.jit(lambda X:jnp.sum(X**2/2,axis=(-2,-1))+E_kin_local(prevoutputs[0].weights,X))
#
#    return estimateobservables.getdefaultprofile().butwith(\
#    p=jax.jit(lambda X:prevoutputs[0].eval(X)**2),\
#    qpratio=lambda X:jnp.ones(X.shape[0],),\
#    observables={'E':total_localenergy},\
#    trueenergies=[ef.totalenergy(5)],\
#    maxiterations=100000)
#batch.genprofile4=genprofile4



if __name__=='__main__':
    cdisplay.session_in_display(batchjob.Batchjob,batch)
