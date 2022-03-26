import util
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import bookkeep as bk




def generate_gamma(f,key,eps=.01,samples=500000):
	covs=jnp.arange(-1+eps,1,eps)
	ones=jnp.ones(covs.size)
	X1,X2=util.correlated_X_pairs(key,ones,covs,samples=samples)
	g=jax.vmap(jnp.vdot,in_axes=(0,0))(f(X1),f(X2))/samples

	covs,g=util.smooth(covs,g,eps)
	dcovs,dg=util.numdiff(covs,g,eps)

	covs,g=util.extend(covs,g,eps,dy=dg)
	dcovs,dg=util.extend(dcovs,dg,eps)

	return {'vals':(covs,g),'dvals':(dcovs,dg),'eps':eps}




ac_name=input('activation: ')

key=jax.random.PRNGKey(0)
gdata=generate_gamma(util.activations[ac_name],key,eps=.01,samples=1000000)

bk.savedata(gdata,'gamma_'+ac_name)
