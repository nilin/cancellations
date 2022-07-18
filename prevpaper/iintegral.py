from jax.config import config
config.update("jax_enable_x64", True)
import jax
import bookkeep as bk
import math
import jax.numpy as jnp
from scipy.io import loadmat
import sys
from util import print_
import matplotlib.pyplot as plt


Fs={'ReLU':lambda p:1/(jnp.sqrt(2*math.pi)*jnp.square(p)),'HS':lambda p:1/(jnp.sqrt(2*math.pi)*p),'tanh':lambda p:jnp.sqrt(math.pi/2)/jnp.sinh(math.pi*p/2)}




def iintegral(n,i,F):
	print_('instance ',i,end='\r')
	s,t,D=bk.get('D/n='+str(n)+'/instance '+str(i))
	ds=s[1]-s[0]
	dt=t[1]-t[0]
	S,T=jnp.meshgrid(s,t)
	I=jnp.linalg.multi_dot([F(t),D,F(s)])*ds*dt
	out=1/(2*math.pi)*I

	print_('','n=',n,' i=',i,' out=',out)
	return out


def loop(nmin,nmax,instances):
	for n in range(nmin,nmax+1):
		print(n)
		for ac in ['ReLU','HS','tanh']:
			print(ac)
			bk.save(jnp.array([iintegral(n,i,Fs[ac]) for i in range(instances)]),'computed_by_iintegral/'+ac+' n='+str(n))


if __name__=='__main__':
	loop(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
	
