import matplotlib.pyplot as plt
import util
import os
import itertools
import bookkeep as bk
import jax.numpy as jnp
import math
import estimates
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
		plt.fill_between(n_squares.keys(),q1,q2,alpha=.3,**kwargs)
		plt.scatter(n_squares.keys(),[jnp.average(s) for _,s in n_squares.items()],s=3,**kwargs)
	else:
		plt.plot(n_squares.keys(),[jnp.average(s) for _,s in n_squares.items()],'o-',**kwargs)
		

	plt.yscale('log')


def get(nmin_,nmax_,seed_,ac_name):
	path='hpcdata/data/range='+nmin_+' '+nmax_+' seed='+seed_+'/'
	n_=range(int(nmin_),int(nmax_)+1)
	return {n:bk.get(path+ac_name+' '+str(n))['outputs'] for n in n_}


if __name__=='__main__':

	kwargs={}

	params=sys.argv[1],sys.argv[2],sys.argv[3]
	if 'bootstrap' in sys.argv:
		kwargs['confidenceinterval']=.995

	#plotsquares(get(*params,'tanh'),color='r',**kwargs)
	plotsquares(get(*params,'ReLU'),color='g',**kwargs)
	plotsquares(get(*params,'HS'),color='b',**kwargs)
#	plotsquares(get(*params,'test'),color='m')

	#plt.plot(range(2,12),[1/x for x in range(2,12)])
	#plt.plot(range(2,12),[1/(x**3) for x in range(2,12)])


	nmin,nmax=int(params[0]),int(params[1])
	n_=range(nmin,nmax+1)

	theta0=lambda n:n
	#theta2=lambda n:1.5*n

	plt.plot(n_,[estimates.proxy1('ReLU',theta0(n)) for n in n_],'g:')
	plt.plot(n_,[estimates.proxy1('HS',theta0(n)) for n in n_],'b:')
	#plt.plot(n_,[estimates.proxy2('ReLU',theta2(n)) for n in n_],'g--')
	#plt.plot(n_,[estimates.proxy2('HS',theta2(n)) for n in n_],'b--')
	plt.savefig('plots/ReLU HS.pdf')

	#plt.savefig('plots/ReLU HS tanh.pdf')
	plt.show()



#def plot_delta(ac_name,keyname='W',**kwargs):
#	nmax=20
#	n_deltas={n:jnp.average(util.mindist(W)) for n in range(nmax) if len(W:=getdata(folder,ac_name,n,keyname))!=0}
#
#	plt.plot(n_deltas.keys(),[d for n,d in n_deltas.items()],**kwargs)
#	plt.yscale('log')

#
#
#def duplicates(L):
#	seen=set()
#	dup=set()
#	for x in L:
#		if x in seen:
#			dup.add(x)
#		seen.add(x)
#	return dup
#			
#
#def verifydistinctkeys(files):
#	keys=[f.split('key=')[-1] for f in files]	
#	dup=duplicates(keys)
#	try:	
#		assert(len(dup)==0)
#	except:
#		raise RuntimeError('randomness keys not distinct: '+str(dup))	
#
#def getdata(folder,ac_name,n,key):
#	files=[folder+f for f in os.listdir('./data/'+folder) if f.startswith(ac_name+' | n='+str(n)+' ')]
#	verifydistinctkeys(files)
#	#return list(itertools.chain(*[bk.getdata(f)[key] for f in files]))
#	data=[bk.getdata(f)[key] for f in files]
#	return jnp.concatenate(data) if len(data)>0 else jnp.array([])
"""

def plot_gamma(key,folder,ac_name):
	key0,*keys=jax.random.split(key,100)
	nmax=20
	data={n:d for n in range(nmax) if len(d:=getdata(folder,'gamma '+ac_name,n))!=0}
	n_squares={n:jnp.array(list(itertools.chain(*d))) for n,d in data.items()}

	plt.plot(n_squares.keys(),[jnp.average(s) for _,s in n_squares.items()],color='r')

	n_s_bootstrap=n_squares

	bootstrapmeans=[util.bootstrapmeans(keys[n],s,resamples=100) for n,s in n_s_bootstrap.items()]
	q1=[jnp.quantile(means,.05) for means in bootstrapmeans]
	q2=[jnp.quantile(means,.95) for means in bootstrapmeans]
	plt.fill_between(n_s_bootstrap.keys(),q1,q2,color='r',alpha=.2)
	plt.yscale('log')

def plot_packing():
	data=bk.getdata('w_packing ReLU')
	n_,normal,gamma=data['range'],data['normal'],data['gamma']
	plt.plot(n_,[gamma[n] for n in n_],'m:')
	plt.plot(n_,[jnp.average(jnp.square(normal[n]))/(1.0*math.factorial(n)) for n in n_],'g:')
"""
