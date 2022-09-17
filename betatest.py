from statistics import harmonic_mean
from cancellations.examples import harmonicoscillator1d,estimateobservables,unsupervised
from cancellations.functions import examplefunctions as ef, functions
from cancellations.display import cdisplay
from cancellations.utilities import numutil, tracking, browse, batchjob, energy, sysutil
import jax.numpy as jnp
import jax
import os









profile=unsupervised.getdefaultprofile()



class Run(cdisplay.Run):
    def execprocess(self):
        unsupervised.execprocess(self)

def run():
    cdisplay.session_in_display(Run,profile)
