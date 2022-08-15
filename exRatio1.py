#
# nilin
# 
# 2022/7
#


import config as cfg
import sys
import jax
import jax.numpy as jnp
import jax.random as rnd
import numpy as np
import AS_functions
import learning
import plottools as pt
import mcmc
import matplotlib.pyplot as plt
import numpy as np
import util
from util import ReLU,DReLU,activations
from jax.numpy import tanh
import pdb
import time
import multivariate as mv
import math
import sys
import dashboard as db
import time
from pynput import keyboard
import testing
import AS_tools
import AS_HEAVY
import examplefunctions
import copy
import AS_functions as ASf
from config import session


#jax.config.update("jax_enable_x64", True)


exname='exRatio1'

explanation='Example '+exname#+': softplus target function'
#timebound=cfg.hour

params={
'ftype':'AS_NN',
'n':5,
'd':1,
'samples_train':100000,
'samples_test':1000,
'fnplotfineness':250,
'widths':[5,10,10,1],
'activation':'tanh',
'weight_decay':.1,
'lossfn':'SI_loss',
'samples_rademacher':100,
'timebound':600,
#'priorities':{'rademachercomplexity':1,'normalization':.01,'normratio':.01},
'priorities':{'rademachercomplexity':1,'normratio':.001,'normalization':.001},
'minibatchsize':50
}


def setparams():
	params.update(cfg.cmdredefs)

	try:
		params['activation']={'r':'ReLU','t':'tanh','d':'DReLU','p':'ptanh'}[cfg.selectone({'r','t','d','p'},cfg.cmdparams)]
	except:
		print(10*'\n'+'Pass target activation function as parameter.\n'+10*'\n')	
		raise Exception

	params['widths'][0]=params['n']

	globals().update(params)
	cfg.varnames=list(params)

	ignore={'plotfineness','minibatchsize','initfromfile','d','checkpoint_interval'}

	sessioninfo=explanation+'\n\nsessionID: '+cfg.sessionID+'\n'+cfg.formatvars([(k,globals()[k]) for k in cfg.varnames],separator='\n',ignore=ignore)
	session.remember('sessioninfo',sessioninfo)


def run(**kwargs):


	if 'targetactivation' in kwargs and 'activation' not in kwargs:
		kwargs['activation']=kwargs['targetactivation']
		kwargs['widths']=kwargs['targetwidths']
		
	globals().update(kwargs)
		
	
	
	cfg.trackduration=True

	
	cfg.outpath='outputs/{}/target={}/{}/'.format(exname,activation,cfg.sessionID)
	cfg.write(session.getval('sessioninfo'),cfg.outpath+'info.txt',mode='w')



	#----------------------------------------------------------------------------------------------------
	

	X=rnd.uniform(cfg.nextkey(),(samples_train,n,d),minval=-1,maxval=1)
	X_test=rnd.uniform(cfg.nextkey(),(samples_test,n,d),minval=-1,maxval=1)


	#----------------------------------------------------------------------------------------------------

	initparams=(ftype,n,d,widths,activation)
	Af=ASf.init_learner(*initparams,lossfns=[])


	#fig0=pt.singlefnplot_all_in_one(X_test,Af.as_static())
	#cfg.savefig('{}{}'.format(cfg.outpath,'Af0.pdf'),fig=fig0)


	#----------------------------------------------------------------------------------------------------
	# round one
	# normalize Af, maximize ratio Af/f, let Af fit random points
	#----------------------------------------------------------------------------------------------------

	lognorm=lambda Y:jnp.log(jnp.average(Y**2))
	norm_one_loss=lambda Y:(jnp.log(jnp.average(Y**2)))**2

	normratiolossgrad=util.combinelossgradfns([mv.gen_lossgrad(Af.NS,lossfn=lognorm),mv.gen_lossgrad(Af.f,lossfn=lognorm)],[1,1],coefficients=[1,-1])
	normalizationlossgrad=mv.gen_lossgrad(Af.f,lossfn=norm_one_loss)	
	rademacherlossgrad=mv.gen_lossgrad(Af.f,lossfn=util.log_SI_loss)

	X_rademacher=rnd.uniform(cfg.nextkey(),(samples_rademacher,n,d),minval=-1,maxval=1)
	#Y_rademacher=rnd.rademacher(cfg.nextkey(),(samples_rademacher,))
	Y_rademacher=rnd.normal(cfg.nextkey(),(samples_rademacher,))

	lossgrad2=util.combinelossgradfns([rademacherlossgrad,normalizationlossgrad,normratiolossgrad],[2,1,1],[priorities[i] for i in ['rademachercomplexity','normalization','normratio']])	
	trainer=learning.Trainer(Af,X_rademacher,Y_rademacher,lossgrad=lossgrad2,weight_decay=weight_decay,minibatchsize=minibatchsize)



	processed=cfg.ActiveMemory()
	display1=Display1(10,cfg.dashboard.width,processed)
	cfg.dashboard.add_display(display1,40,name='bars')
	sc1=cfg.Scheduler(cfg.expsched(50,iterations1))
	sc11=cfg.Scheduler(cfg.expsched(250,iterations1))
	sc2=cfg.Scheduler(cfg.periodicsched(1000,iterations1))
	sc3=cfg.Scheduler(cfg.periodicsched(1000,iterations1))


	cfg.log('starting Af complexity optimization with concurrent normalization, Af/f optimization')



#	fig0=pt.singlefnplot_all_in_one(X_test,Af.as_static())
#	cfg.savefig('{}{}'.format(cfg.outpath,'Af1.pdf'),fig=fig0)


	for i in range(iterations1+1):

		cfg.poke()

		if cfg.mode=='break':
			break
		
		try:

			loss=trainer.step()
			processed.remember('minibatch loss',loss)
			processed.addcontext('minibatch number',i)
			processed.remember('weightnorms',jnp.array([util.norm(l) for l in Af.weights[0]]))

			if sc1.activate(i):
				Af_s=Af.as_static()
				f=Af.get_NS()
				f_s=f.as_static()

				processed.remember('rademacher complexity',jnp.sqrt(1-util.SI_loss(Af_s(X_rademacher),Y_rademacher)))
				processed.remember('Af norm',jnp.average(Af_s(X_test[:100])**2))
				processed.remember('f norm',jnp.average(f_s(X_test[:100])**2))
				processed.compute(['f norm','Af norm'],lambda x,y:x/y,'f/Af')

				del Af_s,f,f_s

				#cfg.print(sys.getsizeof(processed))

				cfg.log('processed')

			if sc11.activate(i):
				plotexample(processed)

			if sc2.activate(i):
				pass

#				plt.close('all')
#				Af_s=Af.as_static()
#				Y=Af_s(X)

#				fig1=pt.singlefnplot_all_in_one(X_test,Af_s)#,Y=Y)
#				cfg.savefig('{}{} minibatches.pdf'.format(cfg.outpath,int(i)),fig=fig1)

			if sc3.activate(i):
				Af_s=Af.as_static()
				Y=Af_s(X)
				data={'X':X,'X_test':X_test,'Y':Y,'Y_test':Af_s(X_test),'sections':pt.CrossSections(X,Y,Af_s)}
				cfg.save(data,cfg.outpath+'XY')	
				del Af_s
				cfg.log('saved data')

					
			

			

		except KeyboardInterrupt:
			break



	cfg.dashboard.del_display('bars')

	cfg.log('saving')

	Af_s=Af.as_static()
	Y=Af_s(X)
	return {'X':X,'X_test':X_test,'Y':Y,'Y_test':Af_s(X_test),'sections':pt.CrossSections(X,Y,Af_s)}



	#--------------------------




class Display1(db.StackedDisplay):

	def __init__(self,height,width,memory):
		super().__init__(height,width,memory)
		
		self.addnumberprint('rademacher complexity',msg='rademacher complexity estimate (in [-1,1]): {:.3}')
		self.addbar('rademacher complexity')
		self.addnumberprint('Af norm',msg='||Af|| = {:.3}')
		self.addnumberprint('f norm',msg='||f|| = {:.3}')
		self.addnumberprint('f/Af',msg='||f||/||Af|| = {:.3}')
		self.addnumberprint('weightnorms',transform=util.norm,msg='weight norm = {:.3}')
		self.addline()
		self.addnumberprint('minibatch number',msg='minibatch number {:.0f}/'+str(iterations1))


def plotexample(memory):

	plt.close('all')
	fig,(ax0,ax1)=plt.subplots(2,1,figsize=(7,9))

	#ax0.plot(*util.swap(*memory.gethist('AS norm','minibatch number')),label='AS norm')
	ax0.plot(*util.swap(*memory.gethist('rademacher complexity','minibatch number')),'m',label='rademacher complexity optimization overlap')

	ax0.plot(*util.swap(*memory.gethist('f norm','minibatch number')),'bo--',label='||f||')
	ax0.plot(*util.swap(*memory.gethist('f/Af','minibatch number')),'rd-',label='||f||/||Af||')
	ax0.legend()
	ax0.set_yscale('log')
	ax0.grid(True,which='major',ls='-',axis='y')
	ax0.grid(True,which='minor',ls=':',axis='y')

	weightnorms,minibatches=memory.gethist('weightnorms','minibatch number')
	for l,layer in enumerate(zip(*weightnorms)):
		ax1.plot(minibatches,layer,label='layer {} weight norm'.format(l+1))
	ax1.legend()
	ax1.set_ylim(bottom=0)
	cfg.savefig('{}{}'.format(cfg.outpath,'weights.pdf'),fig=fig)




	


if __name__=='__main__':

	cfg.dashboard=db.Dashboard0()
	setparams()
	run()
