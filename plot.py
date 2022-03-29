import matplotlib.pyplot as plt
import util
import os
import itertools
import bookkeep as bk
import jax.numpy as jnp
import math
import jax






def duplicates(L):
	seen=set()
	dup=set()
	for x in L:
		if x in seen:
			dup.add(x)
		seen.add(x)
	return dup
			

def verifydistinctkeys(files):
	keys=[f.split('key=')[-1] for f in files]	
	dup=duplicates(keys)
	try:	
		assert(len(dup)==0)
	except:
		raise RuntimeError('randomness keys not distinct: '+str(dup))
	

def getdata(folder,ac_name,n):
	files=[folder+f for f in os.listdir('./data/'+folder) if f.startswith(ac_name+' | n='+str(n)+' ')]
	verifydistinctkeys(files)
	return [bk.getdata(f)['outputs'] for f in files]

def plot(key,folder,ac_name):
	key0,*keys=jax.random.split(key,100)
	nmax=20
	data={n:d for n in range(nmax) if len(d:=getdata(folder,ac_name,n))!=0}
	n_squares={n:jnp.square(jnp.array(list(itertools.chain(*d))))/(1.*math.factorial(n)) for n,d in data.items()}
	n_squares={n:s[:min(500000,s.size)] for n,s in n_squares.items()}

	plt.plot(n_squares.keys(),[jnp.average(s) for _,s in n_squares.items()],color='r')

	#n_s_bootstrap={n:s for n,s in n_squares.items() if s.size<100000}
	n_s_bootstrap=n_squares

	bootstrapmeans=[util.bootstrapmeans(keys[n],s,resamples=100) for n,s in n_s_bootstrap.items()]
	q1=[jnp.quantile(means,.05) for means in bootstrapmeans]
	q2=[jnp.quantile(means,.95) for means in bootstrapmeans]
	plt.fill_between(n_s_bootstrap.keys(),q1,q2,color='b',alpha=.2)
	plt.yscale('log')



if __name__=='__main__':


	key=jax.random.PRNGKey(0)
	plot(key,input('folder '),input('activation function '))
#	plot(key,input('activation function: '))
	plt.show()
