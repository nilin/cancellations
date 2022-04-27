import matplotlib.pyplot as plt
import testing
import jax.numpy as jnp
import pickle


def plotdict(X,Y,*args):
	plt.plot(X,[Y[int(x)] for x in X],*args)

fn='recent'

with open('data multilayer '+fn,'rb') as f:
	data=pickle.load(f)
	antisymmetrized=data['a']
	nonsymmetrized=data['n']

Ls=list(antisymmetrized['ReLU'].keys())

plotdict(Ls,antisymmetrized['tanh'],'r')
plotdict(Ls,antisymmetrized['DReLU'],'g')
plotdict(Ls,antisymmetrized['ReLU'],'b')
#plotdict(Ls,nonsymmetrized['tanh'],'r:')
#plotdict(Ls,nonsymmetrized['DReLU'],'g:')
#plotdict(Ls,nonsymmetrized['ReLU'],'b:')

plt.yscale('log')
plt.savefig('plots multilayer '+fn+'.pdf')


