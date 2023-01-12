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
from cancellations.lossesandnorms import losses,losses2
from cancellations.examples import examples, Barronnorm
from cancellations.examples.Barronnorm import get_barronweight,get_threshold_lg,getBarronfn

from cancellations.config.tracking import log,Profile
from cancellations.run import sampling, template_run


####################################################################################################
#
# Barron norm Ansatz
#
####################################################################################################



class Run(template_run.Game):
    processname='Barron_norm'

    @classmethod
    def getdefaultprofile(cls,**kwargs):
        P=profile=super().getdefaultprofile(**kwargs)
        Barron=Barronnorm.getBarronfn(profile)
        detplayer=examples.getlearner_example(Profile(n=P.n,d=P.d,ndets=P.mdet))
        profile.players={\
            'Ansatz':examples.getlearner_example(Profile(n=P.n,d=P.d,ndets=P.mtarget)),\
            #'Ansatz':harmonicoscillator2d.gettarget(profile),\
            'Barron':Barron,\
            'det':detplayer
            }
        profile.lossgrads=[\
            get_threshold_lg(losses.get_lossgrad_SI(Barron._eval_,profile),.001),\
            get_barronweight(1.0,Barron._eval_,profile),\
            get_threshold_lg(losses.get_lossgrad_SI(detplayer._eval_,profile),.001),\
            #losses.get_lossgrad_SI(detplayer._eval_,profile),\
        ]
        profile.lossnames=['epsB','Barron norm estimate','epsD']
        profile.losstargets=['Barron','Barron','det']
        profile.lossweights=[100.0,.1,100.0,.1]
        profile.sampler=profile.getXYsampler(profile.Xsampler,profile.players['Ansatz'])
        profile.info=str(kwargs)
        return profile

    @classmethod
    def getdefaultprofile2(cls,**kwargs):
        P=profile=super().getdefaultprofile(**kwargs)
        Barron=Barronnorm.getBarronfn(profile)
        detplayer=examples.getlearner_example(Profile(n=P.n,d=P.d,ndets=P.mdet))
        profile.players={\
            'Ansatz':getBarronfn(Profile(n=P.n,d=P.d,m=P.mtarget)),\
            'Barron':Barron,\
            'det':detplayer
            }
        profile.lossgrads=[\
            get_threshold_lg(losses.get_lossgrad_SI(Barron._eval_,profile),.001),\
            get_barronweight(1.0,Barron._eval_,profile),\
            get_threshold_lg(losses.get_lossgrad_SI(detplayer._eval_,profile),.001),\
            #losses.get_lossgrad_SI(detplayer._eval_,profile),\
        ]
        profile.lossnames=['epsB','Barron norm estimate','epsD']
        profile.losstargets=['Barron','Barron','det']
        profile.lossweights=[100.0,.1,100.0,.1]
        profile.sampler=profile.getXYsampler(profile.Xsampler,profile.players['Ansatz'])
        profile.info=str(kwargs)
        return profile

    @classmethod
    def getprofiles(cls):
        return {\
            #'mdet=16, mtarget=1' : partial(cls.getdefaultprofile,n=5,d=2,m=100,mdet=16,mtarget=1),\
            #'mdet=16, mtarget=4' : partial(cls.getdefaultprofile,n=5,d=2,m=100,mdet=16,mtarget=4),\
            #'mdet=16, mtarget=16': partial(cls.getdefaultprofile,n=5,d=2,m=100,mdet=16,mtarget=16),\
            '2-layer mdet=1, mtarget=1' : partial(cls.getdefaultprofile2,n=5,d=2,m=100,mdet=16,mtarget=1),\
            '2-layer mdet=1, mtarget=64': partial(cls.getdefaultprofile2,n=5,d=2,m=100,mdet=16,mtarget=64),\
            '2-layer mdet=16, mtarget=1' : partial(cls.getdefaultprofile2,n=5,d=2,m=100,mdet=16,mtarget=1),\
            '2-layer mdet=16, mtarget=64': partial(cls.getdefaultprofile2,n=5,d=2,m=100,mdet=16,mtarget=64),\
            '2-layer mdet=256, mtarget=1': partial(cls.getdefaultprofile2,n=5,d=2,m=100,mdet=256,mtarget=1),\
            '2-layer mdet=256, mtarget=64': partial(cls.getdefaultprofile2,n=5,d=2,m=100,mdet=256,mtarget=64),\
        }

