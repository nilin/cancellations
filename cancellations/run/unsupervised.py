#
# nilin
# 
# 2022/7
#


import jax.numpy as jnp
from functools import partial
import jax
from jax.tree_util import tree_map

from cancellations.functions import _functions_
from cancellations.functions._functions_ import Product
from cancellations.lossesandnorms import energy
from cancellations.examples import examples

from cancellations.config.tracking import dotdict
from cancellations.run import supervised, sampling, template_run


####################################################################################################
#


class Run_VMC(template_run.Run):
    processname=''

    @classmethod
    def getdefaultprofile(cls):
        profile=supervised.Run.getdefaultprofile().butwith(\
            gettarget=examples.get_harmonic_oscillator2d,\
            samples_train=10**5,\
            weight_decay=0.1)

        profile.initlossgrads=[energy.Kinetic_energy_val_and_grad]
        profile.lossnames=['E']
        profile.proposalfn=sampling.gaussianstepproposal(.1)

        return profile

    def getsampler(run,P):
        q=lambda params,X: run.learner._eval_(params,X)**2
        X0=P._genX_(run.nextkey(),P.batchsize,P.n,P.d)
        return sampling.DynamicSampler(q,P.proposalfn,X0)


