from jax.config import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import bookkeep as bk


def getdata(n,d,i):
	datapath='matlab/Ds/n='+str(n)+'/instance='+str(i+1)+'.mat'
	data=loadmat(datapath)

	thetas,diag,offdiag,W=[np.squeeze(x) for x in [data['ts'],data['diag'],data['offdiag'],data['W']]]
	t_half=2*thetas[jnp.min(jnp.array(jnp.where(diag>.5)))]

	tmax=10*t_half
	keep=jnp.array(jnp.where(thetas<tmax))

	return [jnp.squeeze(x) for x in [thetas[keep],diag[keep],offdiag[keep],t_half,W]]


def array_to_function(y,f_):
	std=y[1]-y[0]
	#@jax.jit
	def f(x):
		weights=jnp.exp(-jnp.square(x[:,:,None]-y[None,None,:])/(2*std**2))
		return jnp.dot(weights,f_)/jnp.sum(weights,axis=-1)
	return f

def array_to_function_logscale(t,f_):
	y=jnp.log(t)
	f_exp=array_to_function(y,f_)

	@jax.jit
	def f(s):
		x=jnp.log(s)
		return f_exp(x)

	return f


def getDtt(n,i):
	thetas,diag,offdiag,t_half,W=getdata(n,3,i)
	diag=array_to_function_logscale(thetas,diag)
	offdiag=array_to_function_logscale(thetas,offdiag)
	mu=jnp.sum(jnp.square(W))/2

	@jax.jit
	def D(s,t):
		positive=s*t>0
		s=jnp.abs(s)
		t=jnp.abs(t)
		avg=jnp.sqrt(s*t)

		Dp=diag(avg)*jnp.exp(-mu*jnp.square(s-t))
		Dm=offdiag(avg)*jnp.exp(-mu*jnp.square(s-t))

		return Dp*positive+Dm*(1-positive)

	return D,t_half
	

def compute_Dtt(D,ds,dt,tmax):
	s=jnp.arange(-tmax,tmax,ds)
	t=jnp.arange(dt/2,tmax,dt)
	S,T=jnp.meshgrid(s,t)
	D=D(S,T)

	t=jnp.concatenate([-jnp.flip(t),t])
	D=jnp.concatenate([jnp.flip(D),D])
	return [s,t,D]

def make_Dtt_and_save(n,i):
	D,t_half=getDtt(n,i)
	tmax=5*t_half
	ds=.2
	dt=.01*tmax
	bk.save(compute_Dtt(D,ds,dt,tmax),'D/n='+str(n)+'/instance '+str(i))


def make_Dtt_and_save_forplot(n,i):
	D,t_half=getDtt(n,i)
	tmax=2*t_half
	ds=.02
	dt=.02
	bk.save(compute_Dtt(D,ds,dt,tmax),'D/forplot/n='+str(n)+'/instance '+str(i))


def loop(nmin,nmax,instances,forplot=False):
	for n in range(nmin,nmax+1):
		print('\nn='+str(n))
		for i in range(instances):
			print('instance '+str(i),end='\r')

			if forplot:	
				make_Dtt_and_save_forplot(n,i)	
			else:
				make_Dtt_and_save(n,i)	


if __name__=='__main__':

	if len(sys.argv)<3:
		print('\npython extend_D_to_plane.py nmin nmax instances\n')
		quit()

	if input('compute for plot? (y/n): ')=='y':	
		forplot=True
	else:	
		forplot=False

	loop(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),forplot)
