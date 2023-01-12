from cancellations.examples import Barronnorm,examples
from cancellations.lossesandnorms import losses
from cancellations.run import template_run
from cancellations.config.batchjob import Batchjob
from cancellations.config import sysutil,tracking
import cancellations.config as cfg
from cancellations.config.tracking import Profile,log
from cancellations.utilities import numutil
import os
import jax.random as rnd, jax.numpy as jnp


class Barron(Barronnorm.Run):
    @staticmethod
    def gettarget(P):
        pass
        
    @classmethod
    def getdefaultprofile(cls,X,Y,rho,**kwargs):
        kwargs['samples_train'],kwargs['n'],kwargs['d']=X.shape
        P=super().getdefaultprofile(X=X,Y=Y,rho=rho,**kwargs)
        P.iterations=1000
        P.finish=lambda run: run.losses
        return P

class Detnorm(template_run.Run_statictarget):
    @staticmethod
    def gettarget(P):
        pass

    @staticmethod
    def getlearner(P):
        return examples.getlearner_example(P)

    @classmethod
    def getdefaultprofile(cls,X,Y,rho,**kwargs):
        kwargs['samples_train'],kwargs['n'],kwargs['d']=X.shape
        P=profile=super().getdefaultprofile(X=X,Y=Y,rho=rho,**kwargs)
        profile.learner=examples.getlearner_example(profile)
        profile.lossgrads=[losses.get_lossgrad_SI(profile.learner._eval_)]
        profile.lossnames=['SI']
        profile.lossweights=[1.0]
        P.iterations=1000
        P.finish=lambda run: run.losses
        return P


class Run(Batchjob):
    def execprocess(self):
        P=self.pickprofile()

        for X,Y,rho,info,ID in P.pipe():
            lossesB=self.run_subprocess(Barron(X=X,Y=Y,rho=rho,m=100,iterations=1000))
            lossesD=self.run_subprocess(Detnorm(X=X,Y=Y,rho=rho,ndets=10,iterations=1000))
            sysutil.write('{} ID={}\n'.format(lossesB,ID),P.outpath_B)
            sysutil.write('{} ID={}\n'.format(lossesD,ID),P.outpath_D)

    @classmethod
    def getdefaultprofile(cls,**kw):
        P=super().getdefaultprofile(**kw)
        pf='outputs/fn_outputs'
        relpaths=[p for p in os.listdir(pf) if 'txt' not in p]
        paths=[os.path.join(pf,fn) for fn in relpaths]
        P.parseinfo=lambda I:'\n'.join(['{}:{}'.format(k,v) for k,v in I.items()])
        P.outpath_B='outputs/comparenorms_Barron.txt'
        P.outpath_D='outputs/comparenorms_det.txt'

        def pipe():
            for i,(p,rp) in enumerate(zip(paths,relpaths)):
                try:
                    X,Y,rho,info=sysutil.load(p)
                    yield (X,Y,rho,info,rp)
                except:
                    pass

        P.pipe=pipe
        return P



#class Genfns(Batchjob):
class Genfns(Batchjob):
    def execprocess(self):
        super().execprocess()
        P=self.pickprofile()
        #P=self.profile=self.getdefaultprofile()
        outpath=P.pf
        for i in range(1,10):

            target=examples.getlearner_example(P.butwith(ndets=i))
            Y=numutil.blockwise_eval(target,blocksize=1000)(P.X)

            fname=os.path.join(outpath,'fn{}'.format(i))
            sysutil.save((P.X,Y,P.rho,''),fname)

    @classmethod
    def getdefaultprofile(cls,n=5,d=3):
        P=profile=Profile(pf='outputs/fn_outputs')
        P.samples_train,P.n,P.d=10**5,n,d
        profile._var_X_distr_=1
        profile._genX_=lambda key,samples,n,d:rnd.normal(key,(samples,n,d))*jnp.sqrt(profile._var_X_distr_)
        profile.X_density=numutil.gen_nd_gaussian_density(var=profile._var_X_distr_)
        profile.X=profile._genX_(rnd.PRNGKey(0),profile.samples_train,profile.n,profile.d)
        profile.rho=profile.X_density(profile.X)
        return P