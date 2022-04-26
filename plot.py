import matplotlib.pyplot as plt
import util
import os
import itertools
import bookkeep as bk
import jax.numpy as jnp
import math
from estimates import *
import jax
import sys


d=3

def plotsquares(data,**kwargs):
	key=jax.random.PRNGKey(0)
	key0,*keys=jax.random.split(key,100)

	n_squares={n:jnp.square(d)/(1.*math.factorial(n)) for n,d in data.items()}


	if 'confidenceinterval' in kwargs:
		bootstrapmeans=[util.bootstrapmeans(keys[n],s,resamples=400) for n,s in n_squares.items()]
		q1=[jnp.quantile(means,1-kwargs['confidenceinterval']) for means in bootstrapmeans]
		q2=[jnp.quantile(means,kwargs['confidenceinterval']) for means in bootstrapmeans]

		kwargs.pop('confidenceinterval')
		plt.fill_between(n_squares.keys(),q1,q2,alpha=.3,**kwargs,lw=.5)
		plt.scatter(n_squares.keys(),[jnp.average(s) for _,s in n_squares.items()],s=3,**kwargs)
	else:
		plt.plot(n_squares.keys(),[jnp.average(s) for _,s in n_squares.items()],'o-',**kwargs)
		#plt.scatter(n_squares.keys(),[jnp.average(s) for _,s in n_squares.items()],**kwargs)
		

	plt.yscale('log')


def get(nmin_,nmax_,seed_,ac_name):
	path='hpcdata/data/range='+nmin_+' '+nmax_+' seed='+seed_+'/'
	n_=range(int(nmin_),int(nmax_)+1)
	return {n:bk.get(path+ac_name+' '+str(n))['outputs'] for n in n_}



if __name__=='__main__':

	kwargs={}
	#kwargs={'s':15}

	estimatechoice=sys.argv[4]+'estimate'

	params=sys.argv[1],sys.argv[2],sys.argv[3]
	path='plots/tanh HS ReLU '+estimatechoice

	if 'bootstrap' in sys.argv:
		kwargs['confidenceinterval']=.995
		path=path+' bootstrap'


	nmin,nmax=1,int(sys.argv[5]); n_=range(nmin,nmax+1)
	#nmin,nmax=int(params[0]),int(params[1]); n_=range(nmin,nmax+1)
	ests={ac_name:jnp.array([globals()[estimatechoice](ac_name,n,d) for n in n_]) for ac_name in ['tanh','ReLU','HS']}
	#n_,ests=geomsmooth(n_,ests,4)

	plt.figure()
	plotsquares(get(*params,'tanh'),color='r',**kwargs)
	plotsquares(get(*params,'ReLU'),color='g',**kwargs)
	plotsquares(get(*params,'HS'),color='b',**kwargs)
	plt.plot(n_,ests['tanh'],'r',lw=1)
	plt.plot(n_,ests['ReLU'],'g',lw=1)
	plt.plot(n_,ests['HS'],'b',lw=1)

	plt.savefig(path+'.pdf')


