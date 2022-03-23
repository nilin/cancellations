import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import matplotlib
import seaborn as sns
import pickle
import time
import bookkeep as bk
import copy
import sys
import jax
import jax.numpy as jnp
import optax
import util
import cancellation as canc

from proxies import *


key=jax.random.PRNGKey(0)
key,*keys=jax.random.split(key,1000)


activationnames=['osc','HS','ReLU','exp','tanh','DReLU']
proxynames=['Z','OP','polyOP','polyZ','OCP','polyOCP','polyOCP_','polyOCP_proxy','extendedpolyOP']
defaultstyles={'OP':'k:','Z':'k-.','polyOP':'r:','polyZ':'r-.','OCP':'k--','polyOCP':'r--','polyOCP_':'r','polyOCP_proxy':'y:','extendedpolyOP':'m'}




def evalproxies(activation,proxychoices,n_,datafolder):
	nmax=max(n_)
	Ws,Xs=[bk.getdata(datafolder+'/WX')[k] for k in ['Ws','Xs']]
	Ws_ordered=bk.getdata(datafolder+'/forplots/W_ordered')
	norms_table={}
	ranges={}
	for proxyname in proxychoices:	
		compnorm=globals()[proxyname+'norm']
		if proxyname=='polyOCP_':
			Ws_=Ws_ordered
			n_=range(3,nmax+1)
		else:
			Ws_=Ws
			n_=range(2,nmax+1)

		norms_table[proxyname]=[compnorm(keys[n],activation,Ws_[n],Xs[n]) for n in n_]
		ranges[proxyname]=n_

	return ranges,norms_table	


"""
color by data, style by estimate
"""
def multiple_activations(ac_name_color,proxy_name_style,datafolder,plotfolder):

	plt.figure()
	plt.yscale('log')
	for ac_name,color in ac_name_color.items():
		try:
			data=bk.getdata(datafolder+'/'+ac_name)
			n_,norms=data['range'],data['norms']
		except:
			print('missing '+ac_name+', skipping')
			continue
		plt.plot(n_,norms,color=color,marker='o',ms=3)

		ranges,norms_table=evalproxies(util.activations[ac_name],proxy_name_style.keys(),n_,datafolder)	
		[plt.plot(ranges[p],norms_table[p],color=color,ls=ls) for p,ls in proxy_name_style.items()]
		
	savename=' '.join(ac_name_color)+' _ '+' '.join(proxy_name_style)
	plt.savefig(plotfolder+'/'+savename+'.pdf')


"""
each estimate a different color and style
"""
def one_plot_per_activation(ac_names,proxychoices,datafolder,plotfolder,**kwargs):

	for ac_name in ac_names:
		plt.figure()
		plt.yscale('log')
		if 'ylim' in kwargs:
			plt.ylim(kwargs.get('ylim'))

		try:
			data=bk.getdata(datafolder+'/'+ac_name)
			n_,norms=data['range'],data['norms']
		except:
			print('missing '+ac_name+', skipping')
			continue
		plt.plot(n_,norms,'bo-')

		n_=range(3,12)
		ranges,norms_table=evalproxies(util.activations[ac_name],proxychoices,n_,datafolder)
		[plt.plot(ranges[p],norms_table[p],defaultstyles[p]) for p in proxychoices]

		savename=ac_name+' _ '+' '.join(proxychoices)
		plt.savefig(plotfolder+'/singledata/'+savename+'.pdf')


def make_colors(l):
	l=list(l)
	palette=sns.color_palette(None,len(l))
	return {l[i]:palette[i] for i in range(len(l))}	



Wtype={'n':'normal','s':'separated','ns':'normal small','ss':'small separated'}[sys.argv[1]]
datafolder=Wtype

plotfolder='plots/'+Wtype
bk.mkdir(plotfolder)
bk.mkdir(plotfolder+'/singledata')



#activations=util.activations
activations={'ReLU':util.ReLU,'HS':util.heaviside,'tanh':jnp.tanh}

### all activation functions in one plot ###
multiple_activations(make_colors(activations.keys()),{'polyOCP':'dotted','polyOCP_':'dashed'},datafolder,plotfolder)

activations=util.activations
multiple_activations(make_colors(activations.keys()),{},datafolder,plotfolder)


#### main proxies ###
one_plot_per_activation(activations.keys(),['polyOP','OCP','polyOCP','polyOCP_'],datafolder,plotfolder)

