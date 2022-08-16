# nilin

import jax
import jax.numpy as jnp
import jax.random as rnd
import util
import config as cfg
#from GPU_sum import sum_perms_multilayer as sumperms
import optax
import math
import testing
import AS_tools
import AS_tools as ASt
import AS_HEAVY
import multivariate as mv
import jax.random as rnd
import examplefunctions
import learning as lrn
import pdb
import backflow as bf




#=======================================================================================================
# antisymmetrized NN
#=======================================================================================================

def initweights_AS_NN(n,d,widths):
	assert widths[0]==n*d and widths[-1]==1, 'widths should be \n[nd...1]\nbut were\n'+str(widths)
	return mv.initweights_NN(widths)


#=======================================================================================================
# backflow+dets
#=======================================================================================================


def initweights_backflowdets(n,d,widths):
	ds=widths[:-1]
	k=widths[-1]
	assert ds[0]==d and ds[-1]==k*n, 'widths should be \n[d...kn,k]\nbut were \n'+str(widths)
	return bf.initweights_backflow(ds)+[util.initweights((k,))]


#==================
# example: ferminet
#==================

def initweights_FN(n,d,widths):
	ds0,ds1,k=widths
	assert ds0[0]==d+1 and ds1[0]==d+ds0[-1] and ds1[-1]==k*n, 'widths should be \n[[d+1...d1],[d+d1...kn],k]\nbut were\n'+str(widths)
	return bf.initweights_FN_backflow([ds0,ds1])+[util.initweights((k,))]



#=======================================================================================================
# Slater sum with NN basis functions
#=======================================================================================================



def initweights_SlaterSumNN(n,d,widths_and_m):
	widths,m=widths_and_m
	assert widths[0]==d, 'Slater NN takes d-dimensional inputs.'
	assert widths[-1]==1, 'Slater NN (separate phis) has 1-dimensional outputs.'
	return [[mv.initweights_NN(widths) for i in range(n)] for k in range(m)]




# init empty learner ----------------------------------------------------------------------------------------------------

def gen_AS_NN(n,d,widths,activation,lossfns=None):
	NN_NS=mv.gen_NN_NS(activation)
	if lossfns==None: lossfns=[cfg.getlossfn()]
	return lrn.AS_Learner(ASt.gen_Af(n,NN_NS),lossgrads=[ASt.gen_lossgrad_Af(n,NN_NS,lossfn) for lossfn in lossfns],NS=NN_NS)

def gen_SlaterSumNN(n,d,widths_and_m,activation):
	NN=mv.gen_NN_wideoutput(activation)
	Af=ASt.gen_SlaterSum(n,NN)
	return lrn.AS_Learner(Af,NS=mv.sum_f(mv.product_f(NN)))

def gen_backflowdets(ac):
	Af=ASt.gen_backflowdets(ac)
	return lrn.AS_Learner(Af)

def gen_FN(ac):
	Af=ASt.gen_FN(ac)
	return lrn.AS_Learner(Af)



# init learner ----------------------------------------------------------------------------------------------------

def init_AS_NN(n,d,widths,activation,**kwargs):
	learner=gen_AS_NN(n,d,widths,activation,**kwargs)
	return learner.reset(initweights_AS_NN(n,d,widths))

def init_SlaterSumNN(n,d,widths_and_m,activation,**kwargs):
	learner=ASt.gen_SlaterSumNN(n,d,widths_and_m,activation,**kwargs)
	return learner.reset(initweights_SlaterSumNN(n,d,widths_and_m))

def init_backflowdets(n,d,widths,ac):
	learner=gen_backflowdets(ac)
	return learner.reset(initweights_backflowdets(n,d,widths))

def init_FN(n,d,widths,ac):
	learner=gen_FN(ac)
	return learner.reset(initweights_FN(n,d,widths))

#----------------------------------------------------------------------------------------------------




def init_learner(learnertype,*args,**kwargs):
	return {\
	'AS_NN':init_AS_NN,\
	'SlaterSumNN':init_SlaterSumNN,
	'backflowdets':init_backflowdets,
	'ferminet':init_FN,
	}[learnertype](*args,**kwargs)



def gen_learner(learnertype,*args,**kwargs):
	return {\
	'AS_NN':gen_AS_NN,\
	'SlaterSumNN':gen_SlaterSumNN,
	}[learnertype](*args,**kwargs)





def init_target(targettype,*args):

	if targettype=='HermiteSlater':
		return examplefunctions.HermiteSlater(args[0],'H',1/8)
	if targettype=='GaussianSlater1D':
		return examplefunctions.GaussianSlater1D(args[0])

	_target_=init_learner(targettype,*args)
	return _target_.as_static(),_target_
		

#=======================================================================================================
# Static target functions
#=======================================================================================================



def init_static(initfn):
	def gen_static_function(*args):
		Af,g_Af,weights=initfn(*args)
		return util.fixparams(Af,weights)
	return gen_static_function


def staticSlater(F):
	return util.noparams(AS_tools.Slater(util.dummyparams(F)))


def Af_from_hist(path,Af):
	weights=cfg.getlastval(path,'weights')
	return util.fixparams(Af,weights)













#---------------------------------------------------------------------------------------------------- 
# tests
#---------------------------------------------------------------------------------------------------- 
"""
#
#def plothermites(n,convention):
#	x=jnp.arange(-4,4,.01)
#	X=jnp.expand_dims(x,axis=-1)
#
#	import matplotlib.pyplot as plt
#
#	F=genhermitefunctions(n,convention)
#	Y=F(x)
#
#	for k in range(n):
#		plt.plot(x,Y[:,k])
#	plt.ylim(-10,10)
#	plt.show()
#
#
#def printpolys(P):
#	for p in P:
#		printpoly(p)
#def printpoly(p):
#	n=p.shape[0]-1
#	pstr=' + '.join([str(p[k])+('' if k==0 else 'x' if k==1 else 'x^'+str(k)) for k in range(n,-1,-1) if p[k]!=0])
#	print(pstr)
#

#
#print(round(hermitecoefficientblock(6,'He')))
#print(round(hermitecoefficientblock(6,'H')))
#
#plothermites(6,'H')
#plothermites(6,'He')

#
#def testSlater():
#	n=5
#	X=rnd.normal(rnd.PRNGKey(0),(10,n,1))
#
#	AS=HermiteSlater(n,'H')
#	testing.verify_antisymmetric(AS,X)
#
#	
"""



if __name__=='__main__':

	pass
	#init_FN()




"""	
#	n,d=4,1
#	widths=[1,3,5]
#	Af=gen_static_SlaterSumNN(n,d,topwidths)
#	testing.verify_antisymmetric(Af,n=4,d=1)
"""
	




