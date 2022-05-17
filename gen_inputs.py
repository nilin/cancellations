import jax
import jax.numpy as jnp
import jax.random as rnd
import bookkeep as bk






def gen_Ws(key,d,n,m_,scaling):
	cxh={'X':1,'H':2}[scaling]
	k1,*ks=rnd.split(key,len(m_)+9)
	return [rnd.normal(k1,(m_[0],n,d))*jnp.sqrt(cxh/(n*d))]+[rnd.normal(ks[l],(m_[l],m_[l-1]))*jnp.sqrt(cxh/m_[l-1]) for l in range(1,len(m_))]
	
def gen_Xs(key,d,n,samples):
	return rnd.normal(key,(samples,n,d))	
	

folder='inputs/'
d=3
instances=100
samples=10000


for n in range(2,26):
	print('\nn='+str(n))
	m=n*d
	for depth in range(2,6):	
		print('depth='+str(depth),end='\r')
		m_=[m]*(depth-1)+[1]
		for scaling in ['X','H']:
			key0,*keys=rnd.split(rnd.PRNGKey(0),instances+10)
			Ws_=[gen_Ws(keys[i],d,n,m_,scaling) for i in range(instances)]
			bk.save(Ws_,folder+'Ws/n='+str(n)+' depth='+str(depth)+' scaling='+str(scaling))
		
	
for n in range(2,26):
	key=rnd.PRNGKey(0)
	Xs_=gen_Xs(key,d,n,samples)
	bk.save(Xs_,folder+'Xs/n='+str(n))



