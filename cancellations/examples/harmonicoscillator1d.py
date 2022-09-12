#
# nilin
# 
# 2022/7
#


import jax
import jax.numpy as jnp
import jax.random as rnd
from . import examplefunctions as ef
from ..functions import functions
from ..functions.functions import ComposedFunction,SingleparticleNN,Product
from ..utilities import arrayutil, config as cfg, tracking, sysutil, textutil
from ..display import cdisplay,display as disp
from . import plottools as pt
from . import exampletemplate

jax.config.update("jax_enable_x64", True)



def getdefaultprofile():
    profile=tracking.Profile(name='run example')
    profile.exname='example'
    profile.instructions=''

    profile.n=5
    profile.d=1

    profile.d_=50
    profile.ndets=10

    profile._X_distr_=lambda key,samples,n,d:rnd.uniform(key,(samples,n,d),minval=-3,maxval=3)
    profile.envelope=jax.jit(lambda X:jnp.all(X**2<1,axis=(-2,-1)))

    # training params

    profile.weight_decay=0
    profile.lossfn=arrayutil.SI_loss
    profile.iterations=25000
    profile.minibatchsize=None

    profile.samples_train=10**5
    profile.samples_test=1000
    profile.evalblocksize=10**4

    profile.adjusttargetsamples=10000
    profile.adjusttargetiterations=250

    profile.act_on_input=exampletemplate.act_on_input
    return profile



def gettarget(profile):
    for i in range(profile.n): setattr(functions,'psi'+str(i),ef.psi(i))
    return functions.Slater(['psi'+str(i) for i in range(profile.n)])

def getlearner(profile):
    d_=profile.d_
    ndets=profile.ndets
    activations=['leakyrelu','leakyrelu','leakyrelu']; d_=50; ndets=10
    return Product(functions.IsoGaussian(1.0),ComposedFunction(\
        SingleparticleNN(widths=[profile.d,50,d_],activation=activations[0]),\
        functions.Backflow(widths=[d_,d_],activation=activations[1]),\
        functions.DetSum(n=profile.n,d=d_,ndets=ndets),\
        functions.OddNN(widths=[1,100,1],activation=activations[2])))



def execprocess(run:tracking.Run):

    run.act_on_input=exampletemplate.act_on_input
    exampletemplate.prepdisplay(run)

    run.outpath='outputs/{}/'.format(run.ID)
    cfg.outpath='outputs/{}/'.format(run.ID)
    tracking.log('imports done')

    
    run.unprocessed=tracking.Memory()
    info='runID: {}\n'.format(run.ID)+'\n'*4; run.trackcurrent('runinfo',info)

    if 'setupdata_path' in run.keys():
        run.update(sysutil.load(run.setupdata_path))
        run.target.restore()
        tracking.log('Loaded target and training data from '+run.setupdata_path)
        info+='target\n\n{}'.format(textutil.indent(run.target.getinfo())); run.trackcurrent('runinfo',info)

    else:
        run.target=gettarget(run)
        info+='target\n\n{}'.format(textutil.indent(run.target.getinfo())); run.trackcurrent('runinfo',info)

        run.X_train=run.genX(run.samples_train)
        run.logcurrenttask('preparing training data')
        run.Y_train=run.target.eval(run.X_train,msg='preparing training data',blocksize=run.evalblocksize)
        run.X_test=run.genX(run.samples_test)
        run.Y_test=run.target.eval(run.X_test,msg='preparing test data',blocksize=run.evalblocksize)
        run.sections=pt.genCrossSections(run.target.eval,interval=jnp.arange(-3,3,6/100))

    run.learner=getlearner(run)
    info+=4*'\n'+'learner\n\n{}'.format(textutil.indent(run.learner.getinfo())); run.trackcurrent('runinfo',info)


    setupdata=dict(X_train=run.X_train,Y_train=run.Y_train,X_test=run.X_test,Y_test=run.Y_test,\
        target=run.target.compress(),learner=run.learner.compress(),sections=run.sections)
    #sysutil.save(setupdata,run.outpath+'data/setup')

    run.trackcurrent('runinfo',info)
    sysutil.write(info,run.outpath+'info.txt',mode='w')

    exampletemplate.testantisymmetry(run.target,run.learner,run.genX(100))
    exampletemplate.train(run,run.learner,run.X_train,run.Y_train,\
        **{k:run[k] for k in ['weight_decay','lossfn','iterations','minibatchsize']})


class Run(tracking.Run):
    execprocess=execprocess
#    def execprocess(self):
#        return execprocess(self)



#
#def main(profile,display):
#    run=tracking.loadprocess(Run(profile,display=display))
#    run.execprocess()
#    tracking.unloadprocess(run)
#


if __name__=='__main__':
    #main(getdefaultprofile(),cfg.session.ID+' default')

    cdisplay.session_in_display(main,getdefaultprofile())
