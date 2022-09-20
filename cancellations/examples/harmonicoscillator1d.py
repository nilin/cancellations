#
# nilin
# 
# 2022/7
#


from re import I
import jax
import jax.numpy as jnp
import jax.random as rnd
from ..functions import examplefunctions as ef, functions
from ..learning import learning
from ..functions.functions import ComposedFunction,SingleparticleNN,Product
from ..utilities import config as cfg, numutil, tracking, sysutil, textutil, sampling, setup
from ..display import cdisplay,display as disp
from . import plottools as pt
from . import exampleutil









class Run(cdisplay.Process):

    processname='harmonicoscillator1d'

    def execprocess(run:cdisplay.Process):
        run.log('imports done')

        P=run.profile
        info='runID: {}\n'.format(run.ID)+'\n'*4; run.infodisplay.msg=info

        # make this temporary
        if 'layernormalization' in P.keys():
            cfg.layernormalization=P.layernormalization 
        if 'initweight_coefficient' in P.keys():
            cfg.initweight_coefficient=P.initweight_coefficient

        run.target=P.gettarget(P)
        info+='target\n\n{}'.format(textutil.indent(run.target.getinfo()))


        run.genX=lambda nsamples: P._genX_(run.nextkey(),nsamples,P.n,P.d)
        run.X_train=run.genX(P.samples_train)
        run.log('preparing training data')
        run.Y_train=numutil.blockwise_eval(run.target,blocksize=P.evalblocksize,msg='preparing training data')(run.X_train)
        run.X_test=run.genX(P.samples_test)
        run.Y_test=numutil.blockwise_eval(run.target,blocksize=P.evalblocksize,msg='preparing test data')(run.X_test)
        r=P.plotrange
        run.sections=pt.genCrossSections(numutil.blockwise_eval(run.target,blocksize=P.evalblocksize),interval=jnp.arange(-r,r,r/50))

        ####

        run.learner=P.getlearner(P)
        info+=4*'\n'+'learner\n\n{}'.format(textutil.indent(run.learner.getinfo()))#; run.infodisplay.msg=info
        info+=10*'\n'+str(run.profile); run.infodisplay.msg=info


        setupdata=dict(X_train=run.X_train,Y_train=run.Y_train,X_test=run.X_test,Y_test=run.Y_test,\
            target=run.target.compress(),\
            learner=run.learner.compress(),\
            sections=run.sections,\
            profilename=P.profilename\
            )
        sysutil.save(setupdata,run.outpath+'data/setup')

        run.unprocessed=tracking.Memory()
        run.unprocessed.target=run.target.compress()

        sysutil.write(info,run.outpath+'info.txt',mode='w')

        #train
        run.lossgrad=gen_lossgrad(run.learner._eval_,P.X_density)

        run.sampler=sampling.SamplesPipe(run.X_train,run.Y_train,minibatchsize=P.minibatchsize)

        run.trainer=learning.Trainer(run.lossgrad,run.learner,run.sampler,\
            **{k:P[k] for k in ['weight_decay','iterations']}) 

        regsched=tracking.Scheduler(tracking.nonsparsesched(P.iterations,start=50))
        plotsched=tracking.Scheduler(tracking.sparsesched(P.iterations,start=1000))
        ld,_=exampleutil.addlearningdisplay(run,run.display)

        run.log('data type (32 or 64): {}'.format(run.learner.eval(run.X_train[100:]).dtype))

        stopwatch1=tracking.Stopwatch()
        stopwatch2=tracking.Stopwatch()

        for i in range(P.iterations+1):

            loss=run.trainer.step()
            for mem in [run.unprocessed,run]:
                mem.minibatchnumber=i
                mem.remember('minibatch loss',loss)

            if regsched.activate(i):
                run.unprocessed.remember('weights',run.learner.weights,minibatchnumber=i)
                run.unprocessed.learner=run.learner.compress()
                sysutil.save(run.unprocessed,run.outpath+'data/unprocessed',echo=False)
                sysutil.write('loss={:.2E} iterations={} n={} d={}'.format(loss,i,P.n,P.d),run.outpath+'metadata.txt',mode='w')	

#            if plotsched.activate(i):
#                exampleutil.fplot()
#                exampleutil.lplot()

            if stopwatch1.tick_after(.05):
                ld.draw()

            if stopwatch2.tick_after(.5):
                if P.act_on_input(setup.checkforinput())=='b': break

        return run.learner


    prepdisplay=exampleutil.prepdisplay

    @staticmethod
    def getdefaultprofile():
        profile=tracking.Profile()
        profile.instructions=''

        profile.gettarget=gettarget
        profile.getlearner=getlearner
        profile.act_on_input=exampleutil.act_on_input

        profile.n=5
        profile.d=1

        profile.learnerparams={\
            'SPNN':dict(widths=[profile.d,25,25],activation='sp'),
            'backflow':dict(widths=[],activation='sp'),
            'dets':dict(d=25,ndets=25),
            #'OddNN':dict(widths=[25,1],activation='sp')
        }

        profile._var_X_distr_=4
        profile._genX_=lambda key,samples,n,d:rnd.normal(key,(samples,n,d))*jnp.sqrt(profile._var_X_distr_)
        profile.X_density=lambda X:jnp.exp(-jnp.sum(X**2/(2*profile._var_X_distr_),axis=(-2,-1)))

        # training params

        profile.weight_decay=0
        profile.iterations=25000
        profile.minibatchsize=100

        profile.samples_train=5*10**4
        profile.samples_test=1000
        profile.evalblocksize=10**4

        profile.adjusttargetsamples=10000
        profile.adjusttargetiterations=250

        profile.plotrange=5

        profile.act_on_input=exampleutil.act_on_input
        return profile


def gen_lossgrad(f,X_density):
    lossfn=lambda params,X,Y: numutil.weighted_SI_loss(f(params,X),Y,relweights=X_density(X))
    return jax.jit(jax.value_and_grad(lossfn))


def gettarget(profile):
    return functions.Slater(*['psi'+str(i) for i in range(profile.n)])



def getlearner(profile):
    return Product(functions.IsoGaussian(1.0),ComposedFunction(\
        SingleparticleNN(**profile.learnerparams['SPNN']),\
        #functions.Backflow(**profile.learnerparams['backflow']),\
        functions.Dets(n=profile.n,**profile.learnerparams['dets']),\
        functions.Sum()\
        ))




