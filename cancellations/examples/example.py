#
# nilin
# 
# 2022/7
#


import jax
import jax.numpy as jnp
import jax.random as rnd
from ..functions import functions
from ..functions.functions import ComposedFunction,SingleparticleNN
from ..utilities import arrayutil, config as cfg, tracking, sysutil, textutil
from ..display import cdisplay,display as disp
from . import plottools as pt
from . import exampletemplate

jax.config.update("jax_enable_x64", True)



def getdefaultprofile():
    profile=tracking.Profile(name='run example')
    profile.exname='example'
    profile.instructions=''

    profile.targetparams={}
    profile.targetchoice='ASNN1'

    profile.learnerparams={}
    profile.learnerchoice='backflow' 


    profile.n=3
    profile.d=2

    profile._X_distr_=lambda key,samples,n,d:rnd.uniform(key,(samples,n,d),minval=-1,maxval=1)
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




def pickexample(choice,n,d,**kw):
    n,d=tracking.pull('n','d')

    match choice:
        # meant as target

        case 'hg':
            return ComposedFunction(functions.Slater('hermitegaussproducts',n=n,d=d,mode='gen'),functions.Outputscaling())
        case 'gauss':
            return ComposedFunction(functions.Slater('parallelgaussians',n=n,d=d,mode='gen'),functions.Outputscaling())
        case 'ASNN1':
            m=10
            return ComposedFunction(functions.ASNN(n=n,d=d,widths=['nd',m,m,1],activation='tanh'),functions.Outputscaling())

        # meant as learner

        case 'slaterNN':
            return ComposedFunction(\
                functions.Slater(SingleparticleNN(widths=[d,100,100,n],activation='tanh')),
                functions.OddNN(widths=[1,100,1],activation='leakyrelu'))

        case 'ASNN2': 
            d_=10
            m=10

            return ComposedFunction(\
                SingleparticleNN(widths=[d,10,10,d_],activation='tanh'),\
                functions.ASNN(n=n,d=d_,widths=['nd',m,1],activation='leakyrelu'),\
                functions.OddNN(widths=[1,10,1],activation='leakyrelu'))

        case 'backflow':
            d_=100
            ndets=10
            activations=['leakyrelu','leakyrelu','leakyrelu']

            return ComposedFunction(\
                SingleparticleNN(widths=[d,100,d_],activation=activations[0]),\
                functions.Backflow(widths=[d_,d_],activation=activations[1]),\
                functions.DetSum(n=n,d=d_,ndets=ndets),\
                functions.OddNN(widths=[1,100,1],activation=activations[2]))


def prep_and_run(run:tracking.Run):

    run.outpath='outputs/{}/'.format(run.ID)
    cfg.outpath='outputs/{}/'.format(run.ID)
    tracking.log('imports done')

    
    run.unprocessed=tracking.Memory()
    info='runID: {}\n'.format(run.ID)+'\n'*4; run.trackcurrent('runinfo',info)

#    if 'loadtarget' in cfg.cmdparams:
#        try: path=cfg.loadtargetpath+'data/setup'
#        except:
#            loadpaths=browse_runs.pickfolders(msg='Choose target from previous run.\n'+\
#                'Only the path in front of the arrow will be used.',\
#                condition=lambda path:os.path.exists(path+'/data/setup'))
#            path=loadpaths[0]+'data/setup'
#        setupdata=cfg.Profile(cfg.load(path))

    if 'setupdata_path' in run.keys():
        run.update(sysutil.load(run.setupdata_path))
        run.target.restore()
        tracking.log('Loaded target and training data from '+run.setupdata_path)
        info+='target\n\n{}'.format(textutil.indent(run.target.getinfo())); run.trackcurrent('runinfo',info)

    else:
        target=pickexample(run.targetchoice,n=run.n,d=run.d,**run.targetparams)

        tracking.log('adjusting target weights')
        exampletemplate.adjustnorms(target,X=run.genX(run.adjusttargetsamples),iterations=run.adjusttargetiterations,learning_rate=.01)
        run.target=target.compose(functions.Flatten(sharpness=1))
        tracking.log('target initialized')

        info+='target\n\n{}'.format(textutil.indent(target.getinfo())); run.trackcurrent('runinfo',info)

        run.X_train=run.genX(run.samples_train)
        run.logcurrenttask('preparing training data')
        run.Y_train=run.target.eval(run.X_train,msg='preparing training data',blocksize=run.evalblocksize)
        run.X_test=run.genX(run.samples_test)
        run.Y_test=run.target.eval(run.X_test,msg='preparing test data',blocksize=run.evalblocksize)
        run.sections=pt.genCrossSections(run.target.eval)

    run.learner=pickexample(run.learnerchoice,n=run.n,d=run.d,**run.learnerparams)
    tracking.log('learner initialized')
    info+=4*'\n'+'learner\n\n{}'.format(textutil.indent(run.learner.getinfo())); run.trackcurrent('runinfo',info)


    setupdata=dict(X_train=run.X_train,Y_train=run.Y_train,X_test=run.X_test,Y_test=run.Y_test,\
        target=run.target.compress(),learner=run.learner.compress(),sections=run.sections)
    sysutil.save(setupdata,run.outpath+'data/setup')

    run.trackcurrent('runinfo',info)
    sysutil.write(info,run.outpath+'info.txt',mode='w')

    exampletemplate.testantisymmetry(run.target,run.learner,run.genX(100))
    exampletemplate.train(run,run.learner,run.X_train,run.Y_train,\
        **{k:run[k] for k in ['weight_decay','lossfn','iterations','minibatchsize']})



def main(profile,display):
    run=tracking.Run(profile,display=display)
    exampletemplate.prepdisplay(run)
    run.act_on_input=exampletemplate.act_on_input
    tracking.loadprocess(run)
    prep_and_run(run)
    tracking.unloadprocess(run)



if __name__=='__main__':
    #main(getdefaultprofile(),cfg.session.ID+' default')

    cdisplay.session_in_display(main,getdefaultprofile())
