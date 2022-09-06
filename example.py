#
# nilin
# 
# 2022/7
#


from re import I
import config as cfg
import functions
import dashboard as db
from config import session
import exampletemplate
import jax
from functions import ComposedFunction,SingleparticleNN
import jax.random as rnd
import browse_runs
import util
import os
jax.config.update("jax_enable_x64", True)




cfg.exname='example'
cfg.instructions='To load and run with previously generated target function (including weights), run\
    \n\n>>python {}.py loadtarget'.format(cfg.exname)
cfg.outpath='outputs/{}/{}/'.format(cfg.exname,cfg.sessionID)


def pickexample(choice,n,d):
    match choice:
        # meant as target

        case 'hg':
            return ComposedFunction(functions.Slater('hermitegaussproducts',n=n,d=d,mode='gen'),functions.Outputscaling())
        case 'gauss':
            return ComposedFunction(functions.Slater('parallelgaussians',n=n,d=d,mode='gen'),functions.Outputscaling())
        case 'ASNN1':
            return ComposedFunction(functions.ASNN(n=n,d=d,widths=['nd',10,10,1],activation='tanh'),functions.Outputscaling())

        # meant as learner

        case 'slaterNN':
            return ComposedFunction(\
                functions.Slater(SingleparticleNN(widths=[d,100,100,n],activation='tanh')),
                functions.OddNN(widths=[1,100,1],activation='leakyrelu'))

        case 'ASNN2': 
            d_=10;
            return ComposedFunction(\
                SingleparticleNN(widths=[d,100,100,d_],activation='tanh'),\
                functions.ASNN(n=n,d=d_,widths=['nd',100,1],activation='leakyrelu'),\
                functions.OddNN(widths=[1,100,1],activation='leakyrelu'))

        case 'backflow':
            d_=100; ndets=10;
            return ComposedFunction(\
                SingleparticleNN(widths=[d,100,d_],activation='leakyrelu'),\
                functions.Backflow(widths=[d_,d_],activation='leakyrelu'),\
                functions.DetSum(n=n,d=d_,ndets=ndets),\
                functions.OddNN(widths=[1,100,1],activation='leakyrelu'))



def prep():
    cfg.log('imports done')
    global n,d

    n=5
    d=2
    targetchoice='ASNN1'
    learnerchoice='backflow'

    cfg.addparams(
    weight_decay=0,
    lossfn='SI_loss',
    samples_train=100000,
    samples_test=1000,
    iterations=10000,
    minibatchsize=None
    )
    cfg.register(globals(),['n','d'])
    cfg.X_distr=lambda key,samples:rnd.uniform(key,(samples,n,d),minval=-1,maxval=1)

    if 'loadtarget' in cfg.cmdparams:
        path=browse_runs.pickfolders(multiple=False,msg='Choose target from previous run.',\
            condition=lambda path:os.path.exists(path+'/data/setup'))+'data/setup'
        target=cfg.load(path)['target']
        target.restore()
        exampletemplate.prepdashboard(cfg.instructions)
    else:
        exampletemplate.prepdashboard(cfg.instructions)
        target=pickexample(targetchoice,n=n,d=d)
        if 'skipadjust' not in cfg.cmdparams:
            cfg.log('adjusting target weights')
            exampletemplate.adjustnorms(target,X=cfg.genX(10000),iterations=500,learning_rate=.01)#,minibatchsize=32)
        target=target.compose(functions.Flatten(sharpness=1))
        cfg.log('target initialized')

    cfg.log('learner initialized')
    learner=pickexample(learnerchoice,n=n,d=d)

    #cfg.dblog(functions.formatinspection(learner.inspect(cfg.genX(55))))

    exampletemplate.testantisymmetry(target,learner,X=cfg.genX(100))

    return target,learner


exampletemplate.runexample(prep)
