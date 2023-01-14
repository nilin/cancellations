from cancellations.examples import Barronnorm
from cancellations.examples import losses
from cancellations.run import runtemplate
import cancellations.functions.examplefunctions as examples
from cancellations.config import sysutil,tracking
import cancellations.config as cfg
from cancellations.config.tracking import Profile,log
from cancellations.utilities import numutil
from functools import partial
from cancellations.config.browse import Browse
import os
import jax
import jax.random as rnd, jax.numpy as jnp


class Barron(Barronnorm.Run):
    @staticmethod
    def gettarget(P):
        pass
        
    @classmethod
    def getdefaultprofile(cls,X,Y,rho,**kwargs):
        kwargs['samples_train'],kwargs['n'],kwargs['d']=X.shape
        P=super().getdefaultprofile(X=X,Y=Y,rho=rho,**kwargs)
        P.finish=lambda run: run.losses
        return P

class Detnorm(runtemplate.Run_statictarget):
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
        #P.iterations=1000
        P.update(**kwargs)
        P.finish=lambda run: run.losses
        return P


class Compare(Batchjob):
    def execprocess(self):
        P=self.pickprofile()
        datapath=self.run_subprocess(Browse(options=os.listdir('outputs')))
        datapath=os.path.join('outputs',datapath)

        relpaths=sorted([p for p in os.listdir(datapath) if 'txt' not in p])
        paths=[os.path.join(datapath,fn) for fn in relpaths]
        P.parseinfo=lambda I:'\n'.join(['{}:{}'.format(k,v) for k,v in I.items()])
        P.outpath_B='outputs/comparenorms_Barron.txt'
        P.outpath_D='outputs/comparenorms_det.txt'

        def pipe():
            for i,(p,rp) in enumerate(zip(paths,relpaths)):
                try:
                    X,Y,rho,plotslice,info=sysutil.load(p)
                    yield (X,Y,rho,plotslice,info,rp)
                except:
                    pass

        P.pipe=pipe
        for X,Y,rho,plotslice,info,ID in P.pipe():
            #breakpoint()
            lossesB=self.run_subprocess(Barron(X=X,Y=Y,rho=rho,m=250,iterations=10**4,batchsize=25,plotslice=plotslice))
            lossesD=self.run_subprocess(Detnorm(X=X,Y=Y,rho=rho,ndets=10,iterations=10**4,batchsize=25,plotslice=plotslice))
            sysutil.write('{} ID={}\n'.format(lossesB,ID),P.outpath_B)
            sysutil.write('{} ID={}\n'.format(lossesD,ID),P.outpath_D)




class Genfns(Batchjob):
    def execprocess(self):
        super().execprocess()
        P=self.pickprofile()
        outpath=P.datapath
        for i in range(10):
            target=P.gentarget(i)
            Y=numutil.blockwise_eval(target,blocksize=1000)(P.X)
            Xplot,I=runtemplate.Run.getslice(P.X,Y,P.rho)
            Yplot=jax.vmap(target.eval)(Xplot)
            fname=os.path.join(outpath,'fn{}'.format(i))
            sysutil.save((P.X,Y,P.rho,(I,Xplot,Yplot),''),fname)

    @classmethod
    def getdefaultprofile(cls,n=5,d=3,datapath='outputs/fn_outputs'):
        P=profile=Profile(datapath=datapath)
        P.samples_train,P.n,P.d=10**6,n,d
        profile.targettype='random'
        profile._var_X_distr_=1
        profile._genX_=lambda key,samples,n,d:rnd.normal(key,(samples,n,d))*jnp.sqrt(profile._var_X_distr_)
        profile.X_density=numutil.gen_nd_gaussian_density(var=profile._var_X_distr_)
        profile.X=profile._genX_(rnd.PRNGKey(0),profile.samples_train,profile.n,profile.d)
        profile.rho=profile.X_density(profile.X)
        return P

    @classmethod
    def getprofiles(cls):
        def getprofile(name):
            match name:
                case 'Harmonic3d':
                    P=cls.getdefaultprofile()
                    P.gentarget=lambda i:examples.get_harmonic_oscillator2d(P,excitation=i)
                case 'Harmonic2d':
                    P=cls.getdefaultprofile(d=2)
                    P.gentarget=lambda i:examples.get_harmonic_oscillator2d(P,excitation=i)
                case 'twolayer':
                    P=cls.getdefaultprofile().butwith(d=2)
                    P.gentarget=lambda i: Barronnorm.getBarronfn(P.butwith(m=10*(i+1)))
            P.datapath=os.path.join('outputs',name)
            return P
        return {
            'Harmonic3d':partial(getprofile,'Harmonic3d'),\
            'Harmonic2d':partial(getprofile,'Harmonic2d'),\
            'twolayer':partial(getprofile,'twolayer'),\
        }
