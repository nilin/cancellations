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
import multivariate as mv
import jax.random as rnd
import pdb





#=======================================================================================================
# antisymmetrized NN
#=======================================================================================================

from multivariate import NN_NS,NN
from AS_tools import gen_Af,gen_lossgrad_Af,gen_SlaterSum_nPhis,gen_SlaterSum_singlePhi
from AS_HEAVY import gen_Af_heavy,gen_lossgrad_Af_heavy,heavy_threshold



def gen_AS_NN(n):
	return AS_tools.gen_Af(n,NN_NS)

def gen_lossgrad_AS_NN(n,lossfn):
	return AS_tools.gen_lossgrad_Af(n,NN_NS,lossfn)


def initweights_AS_NN(n,d,innerwidths,key=cfg.nextkey()):
	if type(innerwidths)!=list:
		cfg.bgtracker.set('event','Casting width to singleton list')
		innerwidths=[innerwidths]
	return mv.initweights_NN([n*d]+innerwidths+[1],key=key)



def init_AS_NN(n,d,innerwidths,key=cfg.nextkey()):
	return gen_AS_NN(n),gen_lossgrad_AS_NN(n,cfg.lossfn),initweights_AS_NN(n,d,innerwidths,key=key)


#=======================================================================================================
# Slater sum with NN basis functions
#=======================================================================================================



def initweights_SlaterSumNN_singlePhi(n,d,topwidths,key=cfg.nextkey()):
	assert topwidths[-1]%n==0, 'NN output needs n equal-sized blocks.'
	return mv.initweights_NN([d]+topwidths,key=key)

def initweights_SlaterSumNN_nPhis(n,d,topwidths,key=cfg.nextkey()):
	_,*keys=rnd.split(key,n+3)
	return [mv.initweights_NN([d]+topwidths,key=keys[i]) for i in range(n)]




	
def init_SlaterSumNN_singlePhi(n,d,topwidths,key=cfg.nextkey()):
	Af=gen_SlaterSum_singlePhi(n,NN)
	return Af,mv.gen_lossgrad(Af),initweights_SlaterSumNN_singlePhi(n,d,topwidths,key=key)
	
def init_SlaterSumNN_nPhis(n,d,topwidths,key=cfg.nextkey()):
	Af=gen_SlaterSum_nPhis(n,NN)
	return Af,mv.gen_lossgrad(Af),initweights_SlaterSumNN_nPhis(n,d,topwidths,key=key)

init_SlaterSumNN=init_SlaterSumNN_singlePhi




#=======================================================================================================
#=======================================================================================================
# Static target functions
#=======================================================================================================
#=======================================================================================================


def make_static(args):
	Af,g_Af,weights=args
	return util.fixparams(Af,weights)



for Af in ['AS_NN','SlaterSumNN','SlaterSumNN_nPhis','SlaterSumNN_singlePhi']:
	def _gen_(*args):
		return make_static(globals()['init_'+Af](*args))	
	globals()['gen_static_'+Af]=_gen_




#----------------------------------------------------------------------------------------------------



def staticSlater(F):
	return util.noparams(AS_tools.Slater(util.dummyparams(F)))








def Af_from_hist(path,Af):
	weights=cfg.getlastval(path,'weights')
	return util.fixparams(Af,weights)







# 
# 
# 
# 	
# 
# #----------------------------------------------------------------------------------------------------
# # Hermite polynomials
# #----------------------------------------------------------------------------------------------------
# 
# 			
# def hermitecoefficients(n,convention):
# 	return He_coefficients(n) if convention=='He' else H_coefficients(n)
# 
# def He_coefficients(n):
# 	if n==0:
# 		return [[1]]
# 	if n==1:
# 		return [[1],[0,1]]
# 	else:
# 		A=He_coefficients(n-1)
# 		a1,a2=A[-1],A[-2]+2*[0]
# 		a=[-(n-1)*a2[0]]
# 		for k in range(1,n+1):
# 			a.append(a1[k-1]-(n-1)*a2[k])
# 		A.append(a)
# 		return A
# 
# def H_coefficients(n):
# 	if n==0:
# 		return [[1]]
# 	else:
# 		A=H_coefficients(n-1)
# 		a1=A[-1]+2*[0]
# 		a=[-a1[1]]
# 		for k in range(1,n+1):
# 			a.append(2*a1[k-1]-(k+1)*a1[k+1])
# 		A.append(a)
# 		return A
# 
# 
# def hermitecoefficientblock(n,convention):
# 	return jnp.array([p+[0]*(n+1-len(p)) for p in hermitecoefficients(n,convention)])
# 
# 
# 
# #----------------------------------------------------------------------------------------------------
# 
# def HermiteSlater(n,convention,envelopevariance):
# 
# 	envelope_singlesample=lambda x:jnp.exp(-jnp.sum(jnp.square(x))/(2*envelopevariance))
# 	envelope=jax.vmap(envelope_singlesample)
# 
# 	@jax.jit
# 	def AF_(X):
# 		AF=staticSlater(genhermitefunctions(n-1,convention))
# 		return envelope(X)*AF(X)
# 
# 	return AF_
# 
# 
# def genhermitefunctions(n,convention):
# 	coefficients=hermitecoefficientblock(n,convention)
# 	return mv.genpolynomialfunctions(coefficients)	
# 
# 
# 
# 





#----------------------------------------------------------------------------------------------------
# other target functions
#----------------------------------------------------------------------------------------------------





"""
#
#def features(X):
#	ones=jnp.ones(X.shape[:-1]+(1,))
#	X_=jnp.concatenate([X,ones],axis=-1)
#
#	secondmoments=X_[:,:,:,None]*X_[:,:,None,:]
#	secondmoments=jnp.triu(secondmoments)
#	return jnp.reshape(secondmoments,X_.shape[:-1]+(-1,))
#
#
#def products(X1,X2):
#	
#	producttable=X1[:,:,:,None]*X2[:,:,None,:]
#	return jnp.reshape(producttable,X1.shape[:-1]+(-1,))
#
#
#def momentfeatures(k):
#
#	def moments(X):
#		ones=jnp.ones(X.shape[:-1]+(1,))
#		X_=jnp.concatenate([X,ones],axis=-1)
#		Y=X_
#		for i in range(k-1):
#			Y=products(Y,X_)
#		return Y
#
#	return moments
#			
#secondmoments=momentfeatures(2)
#
#
#def appendnorm(X):
#	sqnorms=jnp.sum(jnp.square(X),axis=-1)
#	X_=jnp.concatenate([X,sqnorms],axis=-1)
#	return X_
#	
#
#
#
#features=secondmoments
##features=appendnorm
#
#
#def nfeatures(n,d,featuremap):
#	k=rnd.PRNGKey(0)
#	X=rnd.normal(k,(10,n,d))
#	out=featuremap(X)
#	return out.shape[-1],jnp.var(out)
#
#
#class SPfeatures:
#	def __init__(self,key,n,d,m,featuremap):
#		self.featuremap=featuremap
#		d_,var=nfeatures(n,d,featuremap)
#		self.W,self.b=genW_scalaroutput(key,n,d_,m)
#		#self.W,self.b=genW_scalaroutput(key,n,d_,m,randb=True)
#		self.normalization=1/math.sqrt(var)
#
#		
#
#	def evalblock(self,X):
#		F=self.featuremap(X)*self.normalization
#		return sumperms(self.W,self.b,F)
#
#	def eval(self,X,blocksize=250):
#		samples=X.shape[0]
#		blocks=[]
#		#blocksize=250
#		Yblocks=[]
#		a=0
#		while a<samples:
#			b=min(a+blocksize,samples)
#			Yblocks.append(self.evalblock(X[a:b]))
#			a=b
#		return jnp.concatenate(Yblocks,axis=0)
#
#	def evalNS(self,X):
#		F=self.featuremap(X)*self.normalization
#		return nonsym(self.W,self.b,F)
#		
#

"""






#---------------------------------------------------------------------------------------------------- 
# tests
#---------------------------------------------------------------------------------------------------- 


def plothermites(n,convention):
	x=jnp.arange(-4,4,.01)
	X=jnp.expand_dims(x,axis=-1)

	import matplotlib.pyplot as plt

	F=genhermitefunctions(n,convention)
	Y=F(x)

	for k in range(n):
		plt.plot(x,Y[:,k])
	plt.ylim(-10,10)
	plt.show()


def printpolys(P):
	for p in P:
		printpoly(p)
def printpoly(p):
	n=p.shape[0]-1
	pstr=' + '.join([str(p[k])+('' if k==0 else 'x' if k==1 else 'x^'+str(k)) for k in range(n,-1,-1) if p[k]!=0])
	print(pstr)


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




if __name__=='__main__':
	
	n,d=4,1
	topwidths=[3,5]
	Af=gen_static_SlaterSumNN(n,d,topwidths)
	testing.verify_antisymmetric(Af,n=4,d=1)

	




