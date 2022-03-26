import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib
import pickle
import bookkeep as bk
import jax
import jax.numpy as jnp
import util
	








colors={'trivial':'b','separated':'r'}

for ac_name,activation in util.activations.items():
	plt.figure()
	plt.yscale('log')

	for WXname in ['trivial','separated']:
		range1,var=bk.getplotdata(WXname+' '+ac_name)
		print(ac_name)
		print(WXname)
		print(var)
		range2,delta=bk.getplotdata(WXname+' delta')
		plt.plot(range1,var,colors[WXname]+'o-')
		plt.plot(range2,delta,colors[WXname]+':')
		plt.savefig('plots/trivial_vs_separated '+ac_name+'.pdf')

		
plt.figure()
plt.yscale('log')
for WXname in ['trivial','separated']:
	range2,delta=bk.getplotdata(WXname+' delta')
	plt.plot(range2,delta,colors[WXname])
plt.savefig('plots/trivial_vs_separated delta.pdf')
