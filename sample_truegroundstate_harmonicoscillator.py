from statistics import harmonic_mean
from cancellations.examples import harmonicoscillator1d,estimateobservables,profiles as P
from cancellations.functions import examplefunctions as ef, functions
from cancellations.display import cdisplay
from cancellations.utilities import numutil, tracking, browse, batchjob, energy, sysutil
import jax.numpy as jnp
import jax
import os




profile=P.getprofiles('estimateobservables')['true ground state']()


class Run(estimateobservables.Run):
    def execprocess(self):
        sysutil.save(profile.p_descr.compress(),self.outpath+'density')
        sysutil.save(profile.p_descr.compress(),self.outpath+'data/functions/density')
        sysutil.save(profile.psi_descr.compress(),self.outpath+'data/functions/psi')
        super().execprocess()


if __name__=='__main__':
    Run(**profile).run_as_main()
