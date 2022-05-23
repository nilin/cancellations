from jax.config import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import jax.random as rnd
import bookkeep as bk
import sys





def gen_Ws(key,d,n,m_,instances,scaling):
	cxh={'X':1,'H':2}[scaling]
	k1,*ks=rnd.split(key,len(m_)+9)
	return [rnd.normal(k1,(instances,m_[0],n,d))*jnp.sqrt(cxh/(n*d))]+[rnd.normal(ks[l],(instances,m_[l],m_[l-1]))*jnp.sqrt(cxh/m_[l-1]) for l in range(1,len(m_))]
	
def gen_Xs(key,d,n,samples):
	return rnd.normal(key,(samples,n,d))	
	

folder='inputs/'
d=3


#instances=1000
samples=10**5
size=10**5
nmax=16

scaling=sys.argv[1]

for n in range(2,nmax+1):
	key=rnd.PRNGKey(0)
	Xs_=gen_Xs(key,d,n,samples)
	bk.save(Xs_,folder+'Xs/n='+str(n))

for n in range(2,nmax+1):
	print('\nn='+str(n))
	m=n*d
	for depth in range(2,6):	
		print('depth='+str(depth),end='\r')
		m_=[m]*(depth-1)+[1]
		#for scaling in ['X','H']:
		#key0,*keys=rnd.split(rnd.PRNGKey(0),instances+10)
		key0=rnd.PRNGKey(0)
		#Ws_separate=[gen_Ws(keys[i],d,n,m_,1,scaling) for i in range(instances)]
		#bk.save(Ws_separate,folder+'Ws/n='+str(n)+' depth='+str(depth)+' scaling='+str(scaling))

		Ws=gen_Ws(key0,d,n,m_,size,scaling)
		bk.save(Ws,folder+'Ws/n='+str(n)+' depth='+str(depth)+' '+str(scaling))
			
		
	



