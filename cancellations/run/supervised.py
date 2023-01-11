import jax.numpy as jnp
import jax.random as rnd
import os
from os import path
from functools import partial, reduce
from cancellations.config import config as cfg, sysutil, tracking
from cancellations.functions import _functions_
from cancellations.functions._functions_ import ComposedFunction,SingleparticleNN,Product
from cancellations.run import sampling
from cancellations.utilities import numutil, textutil
from cancellations.config.tracking import dotdict
from cancellations.plotting import plotting
from cancellations.display import _display_
from cancellations.lossesandnorms import losses
from cancellations.run import runtemplate
from jax.tree_util import tree_map
import optax


#dontpick


class Run(runtemplate.Run):

    def getsampler(run,P):
        run.log('preparing training data')
        run.target=P.gettarget(P)
        run.info+='target\n\n{}'.format(textutil.indent(run.target.getinfo()))
        sysutil.save(run.target.compress(),path=os.path.join(run.outpath,'data/target'))
        X=P._genX_(run.nextkey(),P.samples_train,P.n,P.d)
        Y=numutil.blockwise_eval(run.target,blocksize=P.evalblocksize,msg='preparing training data')(X)
        Y=normalizeY(Y,P.X_density(X))
        return sampling.SamplesPipe(X,Y,minibatchsize=P.batchsize)

    @staticmethod
    def getdefaultprofile():
        profile=tracking.Profile()
        profile.instructions=''

        #losses
        profile.initlossgrads=[losses.Lossgrad_SI]
        profile.lossnames=['loss']
        profile.lossperiods=[1]*10
        profile.lossweights=[1.0]*10

        profile.n=5
        profile.d=2

        profile.learnerparams=tracking.dotdict(\
            SPNN=dotdict(widths=[profile.d,25,25],activation='sp'),\
            #backflow=dotdict(widths=[],activation='sp'),\
            dets=dotdict(d=25,ndets=25),)
            #'OddNN':dict(widths=[25,1],activation='sp')

        profile._var_X_distr_=1
        profile._genX_=lambda key,samples,n,d:rnd.normal(key,(samples,n,d))*jnp.sqrt(profile._var_X_distr_)
        profile.X_density=numutil.gen_nd_gaussian_density(var=profile._var_X_distr_)

        # training params

        profile.weight_decay=0
        profile.iterations=25000
        profile.batchsize=100

        profile.samples_train=10**5
        profile.samples_test=1000
        profile.evalblocksize=10**4

        profile.adjusttargetsamples=10000
        profile.adjusttargetiterations=250

        profile.plotrange=5


        return profile


def gettarget(P):
    raise NotImplementedError

def normalizeY(Y,rhoX):
    squarednorm=jnp.average(Y**2/rhoX)
    return Y/jnp.sqrt(squarednorm)

def getlearner(profile):
    return Product(_functions_.IsoGaussian(1.0),ComposedFunction(\
        SingleparticleNN(**profile.learnerparams['SPNN']),\
        #functions.Backflow(**profile.learnerparams['backflow']),\
        _functions_.Dets(n=profile.n,**profile.learnerparams['dets']),\
        _functions_.Sum()\
        ))


