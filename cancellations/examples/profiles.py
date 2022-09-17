
from . import harmonicoscillator1d, estimateobservables, unsupervised
from ..functions import examplefunctions as ef, functions
from ..utilities import numutil, energy
import jax.numpy as jnp

def getprofiles(exname):
    profilegenerators=dict()
    pgens=profilegenerators
    match exname:


        case 'harmonicoscillator1d':
            pgens['default']=harmonicoscillator1d.Run.getdefaultprofile


        case 'estimateobservables':

            # 1
            pgens['abstract (non-runnable)']=estimateobservables.Run.getdefaultprofile

            # 2
            def getprofile2():
                psi_descr=harmonicoscillator1d.gettarget(estimateobservables.Run.getdefaultprofile())
                psi=psi_descr.eval
                E_kin_local=numutil.forfixedparams(energy.genlocalkinetic)(psi)
                p_descr=functions.ComposedFunction(psi_descr,'square')
                profile=estimateobservables.Run.getdefaultprofile().butwith(\
                    name='tgsamples',\
                    p=p_descr.eval,\
                    p_descr=p_descr,\
                    qpratio=lambda X:jnp.ones(X.shape[0],),\
                    maxburnsteps=2500,\
                    maxiterations=10**6,\
                    observables={'V':lambda X:jnp.sum(X**2/2,axis=(-2,-1)),'K':E_kin_local},\
                    burn_avg_of=1000)
                profile.trueenergies={k:ef.totalenergy(5)/2 for k in ['V','K']}
                return profile

            pgens['true ground state']=getprofile2


        case 'unsupervised':
            pgens['this example is under development']=harmonicoscillator1d.Run.getdefaultprofile

    return pgens
