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





#=======================================================================================================
# antisymmetrized NN
#=======================================================================================================



def initweights_AS_NN(n,d,widths):
	assert widths[0]==n*d, 'first layer of AS NN should have width n*d'
	assert widths[-1]==1, 'AS NN output layer width should be 1'
	return mv.initweights_NN(widths)



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

def gen_skip_AS_NN(n,d,widths,activation,lossfns=None):
	skip_NN_NS=mv.gen_skip_NN_NS(activation)
	if lossfns==None: lossfns=[cfg.getlossfn()]
	return lrn.AS_Learner(ASt.gen_Af(n,skip_NN_NS),lossgrads=[ASt.gen_lossgrad_Af(n,skip_NN_NS,lossfn) for lossfn in lossfns],NS=skip_NN_NS)

def gen_SlaterSumNN(n,d,widths_and_m,activation):
	NN=mv.gen_NN_wideoutput(activation)
	Af=ASt.gen_SlaterSum(n,NN)
	return lrn.AS_Learner(Af,NS=mv.sum_f(mv.product_f(NN)))



# init learner ----------------------------------------------------------------------------------------------------

def init_AS_NN(n,d,widths,activation,**kwargs):
	learner=gen_AS_NN(n,d,widths,activation,**kwargs)
	return learner.reset(initweights_AS_NN(n,d,widths))

def init_skip_AS_NN(n,d,widths,activation,**kwargs):
	learner=gen_skip_AS_NN(n,d,widths,activation,**kwargs)
	return learner.reset(initweights_AS_NN(n,d,widths))

def init_SlaterSumNN(n,d,widths_and_m,activation,**kwargs):
	learner=ASt.gen_SlaterSumNN(n,d,widths_and_m,activation,**kwargs)
	return learner.reset(initweights_SlaterSumNN(n,d,widths_and_m))


#----------------------------------------------------------------------------------------------------




def init_target(targettype,*args):
	if 'NN' in targettype:
		_target_=init_learner(targettype,*args)
		return _target_.as_static(),_target_
	else:
		return {'HermiteSlater':examplefunctions.HermiteSlater}[targettype](args[0],'H',1/8)
		



def init_learner(learnertype,*args,**kwargs):
	return {\
	'AS_NN':init_AS_NN,\
	'skip_AS_NN':init_skip_AS_NN,\
	'SlaterSumNN':init_SlaterSumNN,
	}[learnertype](*args,**kwargs)



def gen_learner(learnertype,*args,**kwargs):
	return {\
	'AS_NN':gen_AS_NN,\
	'SlaterSumNN':gen_SlaterSumNN,
	}[learnertype](*args,**kwargs)






#=======================================================================================================
#=======================================================================================================
# Static target functions
#=======================================================================================================
#=======================================================================================================



def init_static(initfn):
	def gen_static_function(*args):
		Af,g_Af,weights=initfn(*args)
		return util.fixparams(Af,weights)
	return gen_static_function



#----------------------------------------------------------------------------------------------------



def staticSlater(F):
	return util.noparams(AS_tools.Slater(util.dummyparams(F)))




def Af_from_hist(path,Af):
	weights=cfg.getlastval(path,'weights')
	return util.fixparams(Af,weights)










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
	widths=[1,3,5]
	Af=gen_static_SlaterSumNN(n,d,topwidths)
	testing.verify_antisymmetric(Af,n=4,d=1)

	




