from statistics import harmonic_mean
from cancellations.examples import harmonicoscillator1d,estimateobservables
from cancellations.functions import examplefunctions as ef, functions
from cancellations.display import cdisplay
from cancellations.utilities import numutil, tracking, browse, batchjob, energy, sysutil
import jax.numpy as jnp
import jax
import os





psi_descr=harmonicoscillator1d.gettarget(estimateobservables.Run.getdefaultprofile())
psi=psi_descr.eval
E_kin_local=numutil.forfixedparams(energy.genlocalkinetic)(psi)
p_descr=functions.ComposedFunction(psi_descr,'square')



profile=estimateobservables.Run.getdefaultprofile().butwith(\
    name='tgsamples',\
    p=p_descr.eval,\
    qpratio=lambda X:jnp.ones(X.shape[0],),\
    maxburnsteps=2500,\
    maxiterations=10**6,\
    observables={'V':lambda X:jnp.sum(X**2/2,axis=(-2,-1)),'K':E_kin_local},\
    burn_avg_of=1000)
profile.trueenergies={k:ef.totalenergy(5)/2 for k in ['V','K']}




class Run(estimateobservables.Run):
    def execprocess(self):
        sysutil.save(p_descr.compress(),self.outpath+'density')
        super().execprocess()



if __name__=='__main__':
    Run(**profile).run_as_main()
