import matplotlib.pyplot as plt
import util
import os
import itertools
import bookkeep as bk
import jax.numpy as jnp
import math
import jax




def plot(key,ac_name):
	key0,*keys=jax.random.split(key,100)
	nmax=20
	data={n:d for n in range(nmax) if len(d:=[bk.getdata(f)['outputs'] for f in os.listdir('./data') if f.startswith(ac_name+' | n='+str(n)+' ')])!=0}
	n_squares={n:jnp.square(jnp.array(list(itertools.chain(*d))))/(1.*math.factorial(n)) for n,d in data.items()}

	plt.plot(n_squares.keys(),[jnp.average(s) for _,s in n_squares.items()],color='r')

	n_s_bootstrap={n:s for n,s in n_squares.items() if s.size<10000}

	bootstrapmeans=[util.bootstrapmeans(keys[n],s,resamples=250) for n,s in n_s_bootstrap.items()]
	q1=[jnp.quantile(means,.01) for means in bootstrapmeans]
	q2=[jnp.quantile(means,.99) for means in bootstrapmeans]
	plt.fill_between(n_s_bootstrap.keys(),q1,q2,color='b',alpha=.2)
	plt.yscale('log')



if __name__=='__main__':

	key=jax.random.PRNGKey(0)
	plot(key,input('activation function: '))
	plt.show()
