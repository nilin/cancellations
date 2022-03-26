import matplotlib.pyplot as plt
from seaborn import color_palette
import bookkeep as bk
import sys
import jax
import jax.numpy as jnp
import util

from proxies import *


key=jax.random.PRNGKey(0)
key,*keys=jax.random.split(key,1000)


activationnames=['osc','HS','ReLU','exp','tanh','DReLU']
proxynames=['Z','OP','polyOP','polyZ','OCP','polyOCP','polyOCP_','polyOCP_proxy','extendedpolyOP']
defaultstyles={'OP':'k:','Z':'k-.','polyOP':'r:','polyZ':'r-.','OCP':'k--','polyOCP':'r--','polyOCP_':'r','polyOCP_proxy':'y:','extendedpolyOP':'m','exptaylor':'m:','expapprox':'m--','gamma':'m','gamma_HS_':'m'}




def evalproxies(ac_name,proxychoices,n_,datafolder):
	if len(proxychoices)==0:
		return {},{}
	activation=util.activations[ac_name]
	nmax=max(n_)
	data=bk.getdata(datafolder+'/WX')
	data['Ws_ordered']=bk.getdata(datafolder+'/forplots/W_ordered')
	data['gamma_dists']=bk.getdata('dists_to_pn '+ac_name)

	norms_table={}
	ranges={}
	for proxyname in proxychoices:	
		compnorm=globals()[proxyname+'norm']
		if proxyname=='polyOCP_':
			n_=range(3,nmax+1)
		else:
			n_=range(2,nmax+1)

		norms_table[proxyname]=[compnorm(keys[n],activation,n,data) for n in n_]
		ranges[proxyname]=n_

	return ranges,norms_table	


"""
color by data, style by estimate
"""
def multiple_activations(ac_name_color,proxy_name_style,datafolder,plotfolder):

	print('batch plot '+' '.join(ac_name_color))

	ac_name_color_=dict(ac_name_color)
	plt.figure()
	plt.yscale('log')
	for ac_name,color in ac_name_color.items():
		try:
			data=bk.getdata(datafolder+'/'+ac_name)
			n_,norms=data['range'],data['norms']
		except:
			print('missing '+ac_name+', skipping')
			del ac_name_color_[ac_name]
			continue
		plt.plot(n_,norms,color=color,marker='o',ms=3)

		ranges,norms_table=evalproxies(ac_name,proxy_name_style.keys(),n_,datafolder)	
		ranges_,norms_table_=individualizedproxies(ac_name,n_)
		[plt.plot(ranges[p],norms_table[p],color=color,ls=ls) for p,ls in proxy_name_style.items()]
		[plt.plot(ranges_[p],norms_table_[p],defaultstyles[p]) for p in norms_table_]
		
	savename=' '.join(ac_name_color_)+' _ '+' '.join(proxy_name_style)
	plt.savefig(plotfolder+'/'+savename+'.pdf')

"""
color by data, style by estimate
"""
def activations_and_best_estimate(ac_name_proxy,colors,datafolder,plotfolder):

	print('individual proxies '+' '.join(ac_name_proxy))

	ac_name_proxy_=dict(ac_name_proxy)
	plt.figure()
	plt.yscale('log')
	for ac_name,proxy in ac_name_proxy.items():
		try:
			data=bk.getdata(datafolder+'/'+ac_name)
			n_,norms=data['range'],data['norms']
		except:
			print('missing '+ac_name+', skipping')
			del ac_name_proxy_[ac_name]
			continue
		plt.plot(n_,norms,color=colors[ac_name],marker='o',ms=3)

		ranges,norms_table=evalproxies(ac_name,[proxy],n_,datafolder)	
		plt.plot(ranges[proxy],norms_table[proxy],color=colors[ac_name],ls='dotted')
		
	savename=' '.join([a+'='+p for a,p in ac_name_proxy_.items()])
	plt.savefig(plotfolder+'/'+savename+'.pdf')


"""
each estimate a different color and style
"""
def one_plot_per_activation(ac_names,proxychoices,datafolder,plotfolder,**kwargs):

	print('separate plots '+' '.join(ac_names))

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

		ranges,norms_table=evalproxies(ac_name,proxychoices,n_,datafolder)
		ranges_,norms_table_=individualizedproxies(ac_name,n_)
		[plt.plot(ranges[p],norms_table[p],defaultstyles[p]) for p in norms_table.keys()]
		[plt.plot(ranges_[p],norms_table_[p],defaultstyles[p]) for p in norms_table_.keys()]

		savename=ac_name+' _ '+' '.join(proxychoices)
		plt.savefig(plotfolder+'/singledata/'+savename+'.pdf')

def individualizedproxies(ac_name,n_):
	ranges={}
	norms_table={}
	if ac_name=='exp':
		#ranges['exptaylor']=n_
		#norms_table['exptaylor']=exptaylor(n_)
		ranges['expapprox']=n_
		norms_table['expapprox']=expapprox(n_)
	return ranges,norms_table

def make_colors(l):
	l=list(l)
	palette=color_palette(None,len(l))
	return {l[i]:palette[i] for i in range(len(l))}	



Wtype=util.Wtypes[input('type of W: ')]
datafolder=Wtype

plotfolder='plots/'+Wtype
bk.mkdir(plotfolder)
bk.mkdir(plotfolder+'/singledata')


#activations=util.activations
activations={'ReLU':util.ReLU,'HS':util.heaviside,'tanh':jnp.tanh,'exp':jnp.exp}


ac_proxies={'ReLU':'polyOCP_','HS':'polyOCP','exp':'expapprox'}
activations_and_best_estimate(ac_proxies,make_colors(ac_proxies.keys()),datafolder,plotfolder)

ac_proxies={'ReLU':'polyOCP_','HS':'polyOCP'}
activations_and_best_estimate(ac_proxies,make_colors(ac_proxies.keys()),datafolder,plotfolder)

##### main proxies ###
one_plot_per_activation(util.activations.keys(),['polyOP','OCP','polyOCP'],datafolder,plotfolder)

### all activation functions in one plot ###
multiple_activations(make_colors(activations.keys()),{'polyOCP':'dotted'},datafolder,plotfolder)
multiple_activations(make_colors(activations.keys()),{},datafolder,plotfolder)
multiple_activations(make_colors(util.activations.keys()),{},datafolder,plotfolder)

#comparison
one_plot_per_activation(util.activations.keys(),['polyOCP','polyOCP_'],datafolder,plotfolder)
#one_plot_per_activation(['ReLU','HS'],['polyOCP','polyOCP_proxy'],datafolder,plotfolder)

