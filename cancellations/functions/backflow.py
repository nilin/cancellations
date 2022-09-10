# nilin




import jax.numpy as jnp
import jax
from ..utilities import util, config as cfg
import math
import jax.random as rnd
import pdb

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


def initweights_Backflow(widths,*args,**kw):
	ds=widths
	Ws=[util.initweights((d2,2*d1)) for d1,d2 in zip(ds[:-1],ds[1:])]
	bs=[rnd.normal(cfg.nextkey(),(d2,))*cfg.biasinitsize for d2 in ds[1:]]

	return list(zip(Ws,bs))	




####################################################################################################
# test
####################################################################################################

if __name__=='__main__':

	import testing

	n,d,k=5,3,2
