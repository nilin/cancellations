import gen_outputs
import jax.numpy as jnp
import plot_twolayer
import sys


if len(sys.argv)==1:
	print('\n\nquickgen_twolayerplot.py nmax scaling=X/H\n\n')
	quit()
nmax=int(sys.argv[1])
scaling=sys.argv[2]


acs=['DReLU_normalized','tanh']


for samples in 10**jnp.arange(1,6):
	for depth in [3,4,5]:
		for n in range(2,nmax+1):
			for ac in acs:	
				if n>9 and ac=='exp':
					continue
				gen_outputs.generate_zip(n,n,depth,ac,scaling,samples,'silent')

	
	print('\n\nyou now have data to run \nplot_depths.py '+str(nmax)+' '+scaling+' '+100*'='+'\n\n')

