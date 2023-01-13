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

from os import path
import optax
import copy
from cancellations.config.tracking import log,Profile
from cancellations.config import sysutil, tracking
import cancellations.config as cfg
from cancellations.run import runtemplate, sampling


####################################################################################################
#
# Barron norm Ansatz
#
####################################################################################################



class Run(runtemplate.Run):
    processname='Barron_norm'

    def execprocess(run):

        run.prepdisplay()
        run.log('imports done')

        P=run.profile
        run.info='runID: {}\n'.format(run.ID)+'\n'*4; run.infodisplay.msg=run.info

        run.infodisplay.msg=run.info+P.info
        run.T.draw()

        run.traindata=dict()
        sysutil.write(run.info,path.join(run.outpath,'info.txt'),mode='w')

        run.losses={lossname:None for lossname in P.lossnames}
        run.addlearningdisplay()

        stopwatch1=tracking.Stopwatch()
        stopwatch2=tracking.Stopwatch()

        optimizers=dict()
        for p,player in P.players.items():
            opt=optax.adamw(learning_rate=.01,weight_decay=P.weight_decay)
            state=opt.init(player.weights)
            optimizers[p]=(opt,state)

        for i in range(P.iterations+1):
            run.its=i
            X,*Ys=P.sampler()

            grads={p:[] for p in P.players}

            for lossname,p,lossgrad,weight in zip(P.lossnames,P.losstargets,P.lossgrads,P.lossweights):
                run.losses[lossname],grad=lossgrad(P.players[p].weights,X,*Ys)
                grads[p].append(tree_map(lambda A:weight*A,grad))
            run.traindata[i]=copy.deepcopy(run.losses)            
                
            for p,player in P.players.items():
                if len(grads[p])==0: continue
                opt,state=optimizers[p]
                grad=runtemplate.sumgrads(grads[p])
                updates,state=opt.update(grad,state,player.weights)
                player.weights=optax.apply_updates(player.weights,updates)

            if stopwatch1.tick_after(.05):
                run.learningdisplay.draw()

            if stopwatch2.tick_after(.5):
                if run.act_on_input(cfg.getch(run.getinstructions))=='b': break
        

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

