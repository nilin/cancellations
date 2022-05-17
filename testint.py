import iintegral as iint
import extend_D_to_plane as ext
import matplotlib.pyplot as plt
import bookkeep as bk
import sys
import jax.numpy as jnp


nmin=int(sys.argv[1])
nmax=int(sys.argv[2])
instances=int(sys.argv[3])

if(len(input('enter to compute'))==0):
	ext.loop(nmin,nmax,instances)
iint.loop(nmin,nmax,instances)

n_=range(nmin,nmax)
for n in n_:
	print(bk.get('computed_by_iintegral/ReLU n='+str(n)))

plt.fill_between(n_,[jnp.min(bk.get('computed_by_iintegral/ReLU n='+str(n))) for n in n_],[jnp.max(bk.get('computed_by_iintegral/ReLU n='+str(n))) for n in n_],alpha=.2)
plt.plot(n_,[jnp.median(bk.get('computed_by_iintegral/ReLU n='+str(n))) for n in n_])
plt.yscale('log')
plt.show()
