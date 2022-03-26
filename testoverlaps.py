import permutations
import matplotlib.pyplot as plt
import seaborn as sns
import util
import bookkeep as bk
import math
import jax
import jax.numpy as jnp





#Wtype=util.Wtypes[input('type of W: ')]
Wtype=util.Wtypes['s']
n=int(input('n: '))
datafolder=Wtype
key=jax.random.PRNGKey(0)
key0,*keys=jax.random.split(key,10000)
samples=1000
d=3

Ws=bk.getdata(datafolder+'/WX')['Ws']



def overlaps(n,keys):
	print(n)

	W=Ws[n]
	p_overlaps=[]
	m_overlaps=[]

	for i in range(samples):
		I=jnp.repeat(jnp.eye(n)[:,:,None],3,axis=-1)
		W_=jnp.concatenate([I,W],axis=0)
		pW_=jax.random.permutation(keys[i],W_,axis=-2)
		P=pW_[:n,:,0]
		sign=jnp.linalg.det(P)
		pW=pW_[n:,:,:]
		overlaps=jax.vmap(jnp.vdot)(pW,W)
		if sign>0:
			p_overlaps.append(overlaps[None,:])
		else:
			m_overlaps.append(overlaps[None,:])

	p_overlaps=jnp.concatenate(p_overlaps,axis=0)
	m_overlaps=jnp.concatenate(m_overlaps,axis=0)
	return p_overlaps,m_overlaps

p_overlaps,m_overlaps=overlaps(n,keys)

bw=.01
sns.kdeplot(jnp.ravel(p_overlaps),bw=bw)
sns.kdeplot(jnp.ravel(m_overlaps),bw=bw)
plt.savefig('plots/testoverlaps '+str(n)+'.pdf')
plt.show()

#n_=range(2,21)
#mean_overlaps=[util.L2norm(overlaps(n)) for n in n_]
#	
#plt.plot(n_,mean_overlaps)
#plt.plot(n_,[1/jnp.sqrt(n*d) for n in n_])
#plt.yscale('log')
#plt.savefig('plots/'+Wtype+'/testoverlaps.pdf')
