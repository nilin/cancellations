import gen_outputs
import jax.numpy as jnp
import plot_twolayer
import sys


if sys.argv[1]=='h':
	print('\n\nquickgen_twolayerplot.py nmax scaling=X/H\n\n')
	quit()
nmax=int(sys.argv[1])
scaling=sys.argv[2]


acs=['ReLU','tanh','HS','exp']



for samples in 10**jnp.arange(6):
	for n in range(2,nmax+1):
		for ac in acs:	
			if n>9 and ac=='exp':
				continue
			gen_outputs.generate_zip(n,n,2,ac,scaling,samples,'silent')

	
	print('\n\nyou now have data to run \nplot_twolayer.py '+str(nmax)+' '+scaling+' '+100*'='+'\n\n')

