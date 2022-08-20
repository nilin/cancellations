# nilin




import jax.numpy as jnp
import jax
import util
import math
import jax.random as rnd
import config as cfg
from util import activations
import pdb
import multivariate as mv

jax.config.update("jax_enable_x64", True)


####################################################################################################
#
# general equivariant function
#
####################################################################################################

"""
# phi: params x R^d x R^d -> R^d'
# 
# output F
#
# F: params x R^nd x R^md -> R^nd'
#
# F(W,X,Y) equivariant in X, symmetric in Y
"""
def gen_EV_layer(phi,pool=jnp.sum):
	
	phi_iJ=jax.vmap(phi,in_axes=(None,None,-2),out_axes=-1)
	def pooled_along_y(params,xi,yJ):
		return pool(phi_iJ(params,xi,yJ),axis=-1)

	return jax.jit(jax.vmap(pooled_along_y,in_axes=(None,-2,None),out_axes=-2))




"""
# F(params,PX)=P' F(params,X),
#
# where P' applies P on dimension -2
"""
def gen_backflow(ac):

	NN_layer=mv.gen_NN_layer(ac)
	phi=jax.jit(lambda Wb,x,y:NN_layer(Wb,jnp.concatenate([x,y],axis=-1)))
	layer=gen_EV_layer(phi)
	
	def F(params,Y):
		for Wl in params:
			Y=layer(Wl,Y,Y)	
		return Y
	return jax.jit(F)




def gen_singleparticleNN(activation):
	return jax.vmap(mv.gen_NN_wideoutput(activation),in_axes=(None,-2),out_axes=-2)



####################################################################################################
# initialization
####################################################################################################

def initweights_backflow(ds):

	Ws=[util.initweights((d2,2*d1)) for d1,d2 in zip(ds[:-1],ds[1:])]
	bs=[rnd.normal(cfg.nextkey(),(d2,))*cfg.biasinitsize for d2 in ds[1:]]

	return list(zip(Ws,bs))	






"""
#def norm(X,*args,**kw):
#	return jnp.sqrt(jnp.sum(X**2,*args,**kw))
#
#def dnorm(X):
#	return jnp.expand_dims(norm(X,axis=-1),axis=-1)
"""


####################################################################################################




"""
####################
#
# Example: ferminet
#
####################
# 		
# 
# def gen_FN_preprocess(ac='tanh'):
# 
# 	NN=mv.gen_NN_wideoutput(ac)	
# 
# 	def phi(two_particle_weights,x,y):
# 		stream2=jnp.concatenate([x-y,dnorm(x-y)],axis=-1)
# 		return NN(two_particle_weights,stream2)
# 		
# 	F2p=gen_EV_layer(phi)
# 
# 	def F(two_particle_weights,X):
# 		return jnp.concatenate([X,F2p(two_particle_weights,X,X)],axis=-1)
# 
# 	return jax.jit(F)
# 
# 
# def gen_FN_backflow(ac='tanh'):
# 
# 	F0=gen_FN_preprocess(ac)
# 	F1=gen_backflow(ac)
# 
# 	return util.compose(F0,F1)
# 
# 
# 
# 
# 
# # ds0[0]=d+1 (NN acts on (x-y,|x-y|))
# def initweights_FN_backflow(widths):
# 
# 	ds0,ds1=widths
# 
# 	params0=mv.initweights_NN(ds0)
# 	params1=initweights_backflow(ds1)
# 
# 	return [params0,params1]
# 
"""











####################################################################################################
# test
####################################################################################################

if __name__=='__main__':

	import testing

	n,d,k=5,3,2



	##### test1 #####

	ds=[d,d,d,k*n]
	weights=initweights_backflow(ds)
	BF=gen_backflow('ReLU')
	testing.verify_equivariant(BF,n,d,fixparams=weights)	


	##### test composition #####

	ds0=[d,d,10]
	ds1=[10,10,1]
	weights0=initweights_backflow(ds0)
	weights1=initweights_backflow(ds1)
	BF0=gen_backflow('ReLU')
	BFBF=util.compose(BF0,BF0)
	testing.verify_equivariant(BFBF,n,d,fixparams=[weights0,weights1])	



	##### test3 #####

	ds0=[d+1,d+1,d+1]
	ds1=[2*d+1,2*d+1,k*n]

	weights=initweights_FN_backflow([ds0,ds1])
	FBF=gen_FN_backflow()
	FBF=util.fixparams(FBF,weights)
	testing.verify_equivariant(FBF,n,d)	





	##### test2 #####

	ds0=[d+1,d+1,d+1]

	X=rnd.normal(cfg.nextkey(),(100,n,d))

	weights=mv.initweights_NN(ds0)
	F0=gen_FN_preprocess()



	##### test2 #####

	ds0=[d+1,d+1,d+1]

	weights=mv.initweights_NN(ds0)
	F0=gen_FN_preprocess()
	testing.verify_equivariant(F0,n,d,fixparams=weights)	
