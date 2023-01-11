import jax.numpy as jnp
import jax.random as rnd
import os
from os import path
import copy
from functools import partial, reduce
from cancellations.config import config as cfg, sysutil, tracking
from cancellations.functions import _functions_
from cancellations.functions._functions_ import ComposedFunction,SingleparticleNN,Product
from cancellations.run import sampling
from cancellations.utilities import numutil, textutil
from cancellations.config.tracking import dotdict
from cancellations.plotting import plotting
from cancellations.display import _display_
from cancellations.lossesandnorms import losses
from cancellations.run import runtemplate
from cancellations.run.runtemplate import sumgrads, getlearner_example
from jax.tree_util import tree_map
import optax


class Run(runtemplate.Run):

    processname='game template'
    processtype='runs'

    def execprocess(run):

        run.prepdisplay()
        run.log('imports done')

        P=run.profile
        run.info='runID: {}\n'.format(run.ID)+'\n'*4; run.infodisplay.msg=run.info

        run.infodisplay.msg=run.info+P.info
        run.T.draw()

        run.traindata=dict()
        sysutil.write(run.info,path.join(run.outpath,'info.txt'),mode='w')

        regsched=tracking.Scheduler(range(0,P.iterations+25,25))
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
            X,*Ys=P.sampler()

            grads={p:[] for p in P.players}

            for lossname,p,lossgrad,weight in zip(P.lossnames,P.losstargets,P.lossgrads,P.lossweights):
                run.losses[lossname],grad=lossgrad(P.players[p].weights,X,*Ys)
                grads[p].append(tree_map(lambda A:weight*A,grad))
            run.traindata[i]=copy.deepcopy(run.losses)            
                
            for p,player in P.players.items():
                if len(grads[p])==0: continue
                opt,state=optimizers[p]
                grad=sumgrads(grads[p])
                updates,state=opt.update(grad,state,player.weights)
                player.weights=optax.apply_updates(player.weights,updates)

            run.its=i
            #if regsched.activate(i):
            #    run.traindata[i]['weights']=run.learner.weights
            #    sysutil.save(run.learner.compress(),path=path.join(run.outpath,'data','learner'))
            #    sysutil.save(run.traindata,path.join(run.outpath,'data','traindata'),echo=False)
            #    sysutil.write('loss={:.2E} iterations={} n={} d={}'.format(loss,i,P.n,P.d),path.join(run.outpath,'metadata.txt'),mode='w')    

            if stopwatch1.tick_after(.05):
                run.learningdisplay.draw()

            if stopwatch2.tick_after(.5):
                if run.act_on_input(cfg.getch(run.getinstructions))=='b': break
        
        plotting.allplots(run)
        return run.learner
