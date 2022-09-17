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
from ..utilities import config as cfg, numutil, tracking, sysutil, textutil, sampling, energy
from ..display import cdisplay,display as disp
from . import plottools as pt
from . import exampletemplate

jax.config.update("jax_enable_x64", True)



def getdefaultprofile():
    profile=tracking.Profile(name='unsupervised')
    profile.exname='example'
    profile.instructions=''

    profile.n=5
    profile.d=1

    profile.learnerparams={\
        'SPNN':dict(widths=[profile.d,25,25],activation='sp'),
        'backflow':dict(widths=[],activation='sp'),
        'dets':dict(d=25,ndets=25),
        'OddNN':dict(widths=[25,1],activation='sp')
    }

    profile._var_X_distr_=4
    profile._X_distr_=lambda key,samples,n,d:rnd.normal(key,(samples,n,d))*jnp.sqrt(profile._var_X_distr_)
    profile._X_distr_density_=lambda X:jnp.exp(-jnp.sum(X**2/(2*profile._var_X_distr_),axis=(-2,-1)))

    # training params

    profile.weight_decay=0
    profile.iterations=25000
    profile.minibatchsize=100

    profile.samples_train=5*10**4
    profile.samples_test=1000
    profile.evalblocksize=10**4

    profile.adjusttargetsamples=10000
    profile.adjusttargetiterations=250

    profile.act_on_input=exampletemplate.act_on_input
    return profile



#def gettarget(profile):
#    #for i in range(profile.n): setattr(functions,'psi'+str(i),ef.psi(i))
#    #return ComposedFunction(functions.Slater(*['psi'+str(i) for i in range(profile.n)]),functions.Outputscaling())
#    return functions.Slater(*['psi'+str(i) for i in range(profile.n)])

def getlearner(profile):

    return Product(functions.IsoGaussian(1.0),ComposedFunction(\
        SingleparticleNN(**profile.learnerparams['SPNN']),\
        #functions.Backflow(**profile.learnerparams['backflow']),\
        functions.Dets(n=profile.n,**profile.learnerparams['dets']),\
        functions.OddNN(**profile.learnerparams['OddNN'])))




def execprocess(run:cdisplay.Run):

    run.act_on_input=exampletemplate.act_on_input
    exampletemplate.prepdisplay(run)

    run.outpath='outputs/{}/'.format(run.ID)
    cfg.outpath='outputs/{}/'.format(run.ID)
    tracking.log('imports done')

    
    info='runID: {}\n'.format(run.ID)+'\n'*4; run.infodisplay.msg=info

    run.X_train=run.genX(run.samples_train)
    run.logcurrenttask('preparing training data')
    run.X_test=run.genX(run.samples_test)
    #r=5
    #run.sections=pt.genCrossSections(numutil.blockwise_eval(run.target,blocksize=run.evalblocksize),interval=jnp.arange(-r,r,r/50))

    run.learner=getlearner(run)
    info+=4*'\n'+'learner\n\n{}'.format(textutil.indent(run.learner.getinfo())); run.infodisplay.msg=info



    setupdata=dict(X_train=run.X_train,X_test=run.X_test,\
        learner=run.learner.compress(),
        sections=run.sections)
    sysutil.save(setupdata,run.outpath+'data/setup')

    run.unprocessed=tracking.Memory()

    run.trackcurrent('runinfo',info)
    sysutil.write(info,run.outpath+'info.txt',mode='w')

    #train
    run.lossgrad=gen_lossgrad(run.learner._eval_)

    run.trainer=learning.Trainer(run.lossgrad,run.learner,run.X_train,\
        memory=run,**{k:run[k] for k in ['weight_decay','iterations','minibatchsize']}) 

    regsched=tracking.Scheduler(tracking.nonsparsesched(run.iterations,start=100))
    plotsched=tracking.Scheduler(tracking.sparsesched(run.iterations,start=1000))
    run.trainer.prepnextepoch(permute=False)
    ld,_=exampletemplate.addlearningdisplay(run,tracking.currentprocess().display)

    stopwatch1=tracking.Stopwatch()
    stopwatch2=tracking.Stopwatch()

    for i in range(run.iterations+1):

        run.trainer.step()
        for mem in [run.unprocessed,run]:
            mem.addcontext('minibatchnumber',i)
            mem.remember('minibatch loss',loss)

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


def gen_lossgrad(psi):
    V=lambda X:jnp.sum(X**2/2,axis=(-2,-1))
    return energy.gen_logenergy_grad(energy.genlocalenergy(psi,V),lambda params,X:psi(params,X)**2)

#from inspect import signature



class Run(cdisplay.Run):
    execprocess=execprocess
