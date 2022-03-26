import math
import util
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import bookkeep as bk


#cut=lambda x:jnp.max(jnp.min(x,1),-1)
cut=jnp.tanh

ac_name=input('activation: ')

key=jax.random.PRNGKey(0)
key0,*keys=jax.random.split(key,10000)

x_range,y=bk.getdata('gamma_'+ac_name)['vals']
f=util.listasfunction(x_range,y)

def dists_to_p(n,r,f,key,samples=10000):
	X=jax.random.normal(key,(samples,))
	X=cut(r*X)
	Y=f(X)
	a,dist=util.polyfit(X,Y,n)
	return dist
	
	
loginterval=jnp.arange(-10,0,.1)
i_=range(loginterval.size)
dists_to_polynomials=[{'log_r':jnp.array(loginterval),'L2dist':jnp.array([dists_to_p(n,jnp.exp(loginterval[i]),f,keys[20*i+n]) for i in i_])} for n in range(0,20)]

bk.savedata(dists_to_polynomials,'dists_to_pn '+ac_name)


plt.plot(x_range,y)
if(ac_name=='HS'):
	x_=jnp.array(x_range)
	plt.plot(x_,jnp.arctan(jnp.sqrt((x_+1)/(-x_+1)))/math.pi)

plt.savefig('plots/gamma '+ac_name+'.pdf')

plt.figure()
plt.yscale('log')
plt.xscale('log')
for n in range(10):
	graph=dists_to_polynomials[n]
	plt.plot(jnp.exp(graph['log_r']),graph['L2dist'])

plt.savefig('plots/dist_to_pn '+ac_name+'.pdf')


