#
# nilin
# 
# 2022/7
#


from re import I
import jax
import jax.numpy as jnp
import jax.random as rnd

from ...examples import energy

from .. import sampling

from ...config import config as cfg, sysutil, tracking
from ...functions import _functions_, examplefunctions as ef
from ...functions._functions_ import ComposedFunction,SingleparticleNN,Product
from ...utilities import numutil, textutil
import optax
from ...plotting import plottools as pt
from ...examples import exampleutil




def getdefaultprofile():
    profile=tracking.Profile(name='unsupervised')
    profile.exname='example'
    profile.instructions=''

    profile.n=3
    profile.d=1

    profile.learnerparams={\
        'SPNN':dict(widths=[profile.d,25,25],activation='sp'),
        'backflow':dict(widths=[],activation='sp'),
        'dets':dict(d=25,ndets=25),
        'OddNN':dict(widths=[25,1],activation='sp')
    }

    profile._var_X0_distr_=4
    profile._X0_distr_=lambda key,samples,n,d:rnd.normal(key,(samples,n,d))*jnp.sqrt(profile._var_X0_distr_)
    profile._X0_distr_density_=lambda X:jnp.exp(-jnp.sum(X**2/(2*profile._var_X_distr_),axis=(-2,-1)))
    profile.proposalfn=sampling.gaussianstepproposal(.1)

    # training params

    profile.nrunners=100
    profile.burnsteps=1000
    profile.thinning=1
    profile.learningrate=.001

    profile.weight_decay=0
    profile.iterations=25000
    profile.minibatchsize=100

    profile.adjusttargetsamples=10000
    profile.adjusttargetiterations=250

    profile.act_on_input=exampleutil.act_on_input

    profile.truevalue=ef.totalenergy(profile.n)
    return profile


def getlearner(profile):

    return Product(_functions_.IsoGaussian(1.0),ComposedFunction(\
        SingleparticleNN(**profile.learnerparams['SPNN']),\
        #functions.Backflow(**profile.learnerparams['backflow']),\
        _functions_.Dets(n=profile.n,**profile.learnerparams['dets']),\
        _functions_.OddNN(**profile.learnerparams['OddNN'])))



class Run(cdisplay.Process):
    exname='unsupervised'

    def execprocess(run:cdisplay.Process):

        run.act_on_input=exampleutil.act_on_input
        exampleutil.prepdisplay(run)

        run.outpath='outputs/{}/'.format(run.ID)
        cfg.outpath='outputs/{}/'.format(run.ID)
        tracking.log('imports done')

        
        info='runID: {}\n'.format(run.ID)+'\n'*4; run.infodisplay.msg=info

        run.learner=getlearner(run)
        info+=4*'\n'+'learner\n\n{}'.format(textutil.indent(run.learner.getinfo())); run.infodisplay.msg=info


        setupdata=dict(\
            learner=run.learner.compress(),
            sections=run.sections)
        sysutil.save(setupdata,run.outpath+'data/setup')

        run.unprocessed=tracking.Memory()

        run.trackcurrent('runinfo',info)
        sysutil.write(info,run.outpath+'info.txt',mode='w')


        X0=run._X0_distr_(tracking.nextkey(),run.nrunners,run.n,run.d)
        run._density_=lambda params,X: run.learner._eval_(params,X)**2
        sampler=sampling.DynamicSampler(run._density_,run.proposalfn,X0)
        tracking.log('burning')
        for i in range(run.burnsteps):
            sampler.step(run.learner.weights)
        tracking.log('burning done')
        #train
        run.lossgrad=gen_lossgrad(run.learner._eval_)

        run.trainer=TestTrainer(run.lossgrad,run.learner,sampler,\
            **{k:run[k] for k in ['weight_decay','iterations']}) 

        run.trainer.thinning=run.thinning
        run.trainer.learningrate=run.learningrate

        regsched=tracking.Scheduler(tracking.nonsparsesched(run.iterations,start=100))
        plotsched=tracking.Scheduler(tracking.sparsesched(run.iterations,start=1000))
        ld,_=addlearningdisplay(run,tracking.currentprocess().display)

        stopwatch1=tracking.Stopwatch()
        stopwatch2=tracking.Stopwatch()

        for i in range(run.iterations+1):

            loss=run.trainer.step()
    
            for mem in [run.unprocessed,run]:
                mem.addcontext('minibatchnumber',i)
                mem.remember('energy',jnp.exp(loss))

            if regsched.activate(i):
                run.unprocessed.remember('weights',run.learner.weights)
                run.unprocessed.learner=run.learner.compress()
                sysutil.save(run.unprocessed,run.outpath+'data/unprocessed',echo=False)
                sysutil.write('loss={:.3f} iterations={} n={} d={}'.format(loss,i,run.n,run.d),run.outpath+'metadata.txt',mode='w')    

    #        if plotsched.activate(i):
    #            exampletemplate.fplot()
    #            exampletemplate.lplot()

            if stopwatch1.tick_after(.05):
                ld.draw()

            if stopwatch2.tick_after(.5):
                if tracking.act_on_input(tracking.checkforinput())=='b': break

        return run.learner

    getdefaultprofile=getdefaultprofile


def gen_lossgrad(psi):
    V=lambda X:jnp.sum(X**2/2,axis=(-2,-1))
    return energy.gen_logenergy_grad(energy.genlocalenergy(psi,V),lambda params,X:psi(params,X)**2)



class TestTrainer(learning.Trainer):

    def minibatch_step(self,X_mini):

        loss,grad=self.lossgrad(self.learner.weights,X_mini)
        #updates,self.state=self.opt.update(grad,self.state,self.learner.weights)
        #self.learner.weights=optax.apply_updates(self.learner.weights,updates)
        self.learner.weight=numutil.leafwise(lambda x,y:x-self.learningrate*y,self.learner.weights,grad)
        return loss

    def step(self):
        X_mini=jnp.concatenate([self.sampler.step(self.learner.weights) for i in range(self.thinning)],axis=0)
        return self.minibatch_step(X_mini)    






def addlearningdisplay(run,display):

    a,b=display.xlim[0]+2,display.xlim[1]-2

    ld=cdisplay.ConcreteStackedDisplay((a,b),(display.height-10,display.height-1))
    ld.add(disp.NumberPrint('energy',msg='energy estimate {:.2E}',avg_of=100))

    ld.add(disp.Range(run.getqueryfn('energy'),run.truevalue,1))
    ld.add(disp.VSpace(1))
    ld.add(disp.NumberPrint('minibatchnumber',msg='minibatch number {:.0f}'))

    return run.display.add(ld,'learningdisplay')



#dontpick