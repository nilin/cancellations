#
# nilin
# 
# 2022/7
#


import jax.numpy as jnp
from functools import partial

from cancellations.functions import _functions_
from cancellations.functions._functions_ import Product
from cancellations.lossesandnorms import losses,losses2
from cancellations.run import supervised



class Run(supervised.Run):
    processname='harmonicoscillator2d'

    @classmethod
    def getdefaultprofile(cls):
        return super().getdefaultprofile().butwith(gettarget=gettarget,samples_train=10**5)

    @classmethod
    def getprofiles(cls):
        profiles=dict()
        default=cls.getdefaultprofile().butwith(n=6,weight_decay=.1,iterations=10**4)
        profiles['n=6 d=2 SI']=default
        profiles['n=6 d=2 non-SI']=default.butwith(initlossgrads=[losses.Lossgrad_nonSI])
#        profiles['n=6 d=2 bias-reduced SI loss']=\
#            {\
#                'balanced':default.butwith(\
#                    initlossgrads=[partial(losses.Lossgrad_balanced,100,10,mode='nonsquare',batchmode='batch')],\
#                    batchsize=100),\
#                'balanced, control normratio':default.butwith(\
#                    initlossgrads=[partial(losses.Lossgrad_balanced,100,10,mode='nonsquare',batchmode='batch'),\
#                            losses.Lossgrad_normratio],\
#                    lossnames=['loss','normratio_loss'],\
#                    lossperiods=[1,250],\
#                    batchsize=100),\
#                'balanced, NO WEIGHT DECAY':default.butwith(weight_decay=0.0,\
#                    initlossgrads=[partial(losses.Lossgrad_balanced,100,10,mode='nonsquare',batchmode='batch')],\
#                    batchsize=100),\
#                'balanced, NO WEIGHT DECAY, control ratio':default.butwith(weight_decay=0.0,\
#                    initlossgrads=[partial(losses.Lossgrad_balanced,100,10,mode='nonsquare',batchmode='batch'),\
#                        losses.Lossgrad_normratio],\
#                    lossnames=['loss','normratio_loss'],\
#                    lossperiods=[1,250],\
#                    batchsize=100),\
#                'balanced, NO WEIGHT DECAY, decrease |Af|':default.butwith(weight_decay=0.0,\
#                    initlossgrads=[partial(losses.Lossgrad_balanced,100,10,mode='nonsquare',batchmode='batch'),\
#                        partial(losses.Lossgrad_normratio,Afpow=2,fpow=0)],\
#                    lossnames=['loss','normratio_loss'],\
#                    lossperiods=[1,250],\
#                    batchsize=100),\
##                'balanced, squared, small mb, batch 100':default.butwith(\
##                    initlossgrads=[partial(losses.Lossgrad_balanced,100,10,mode='square',batchmode='batch')],\
##                    batchsize=1000),\
##                'balanced, hopeforthebest, small mb, batch 100':default.butwith(\
##                    initlossgrads=[partial(losses.Lossgrad_balanced,100,10,mode='hopeforthebest',batchmode='batch')],\
##                    batchsize=1000),\
#                'separate denominators':default.butwith(\
#                    initlossgrads=[partial(losses2.Lossgrad_separate_denominators,100,10,batchmode='batch')],\
#                    batchsize=100),\
#                'separate denominators, control normratio':default.butwith(\
#                    initlossgrads=[partial(losses2.Lossgrad_separate_denominators,100,10,batchmode='batch'),\
#                            losses.Lossgrad_normratio],\
#                    lossnames=['loss','normratio_loss'],\
#                    lossperiods=[1,250],\
#                    batchsize=100),\
#                'small mb, reference (biased)':default.butwith(\
#                    batchsize=10),\
#            }
        return profiles

def gettarget(P):
    return _functions_.Slater(*['psi{}_{}d'.format(i,P.d) for i in range(1,P.n+1)])



