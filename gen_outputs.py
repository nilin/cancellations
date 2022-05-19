import jax
import jax.numpy as jnp
import GPU_sum
import bookkeep as bk
import math
import testing
import util
from util import print_
from util import str_
import sys
import os


def split_data(Xs,mode='instance'):
	samples,n,d=Xs.shape
	batchsize=max(round(10000/math.factorial(n)),2)
	start=0
	batches=[]
	indices=[]
	while start<samples:
		end=min(start+batchsize,samples)
		batches.append(jnp.expand_dims(Xs[start:end],axis=(1 if mode=='zip' else 0)))
		indices.append(range(start,end))
		start=end
	return batches,indices

def split_zip(Ws,Xs):
	Xbatches,indices=split_data(Xs,'zip')
	Ls=range(len(Ws[0]))


	Wbatches=[[jnp.concatenate([Ws[i][l] for i in batch],axis=0) for l in Ls] for batch in indices]
	return Wbatches,Xbatches


def getdata(n,depth,scaling,instances,samples):
	Ws=bk.get('inputs/Ws/n='+str(n)+' depth='+str(depth)+' scaling='+scaling)
	Xs=bk.get('inputs/Xs/n='+str(n))
	
	Ws=Ws[:min(len(Ws),instances)]
	Xs=Xs[:min(Xs.shape[0],samples)]

	return Ws,Xs


"""
"""


def generate(nmin,nmax,depth,ac_name,scaling,instances,samples,mode='standard'):

	NN_nd=testing.get_NN_nd(ac_name)
	for n in range(nmin,nmax+1):
		fn='outputs/depth='+str(depth)+' NS/'+ac_name+' n='+str(n)+' scaling='+scaling
		if os.path.isfile(fn):
			continue			
		networks,Xs=getdata(n,depth,scaling,100,1000)
		NS=jnp.stack([NN_nd([W[0] for W in network],Xs) for network in networks],axis=0)
		bk.save(NS,fn)
		print_('nonsymmetrized n='+str(n),mode,end='\r')

	for n in range(nmin,nmax+1):
		print(ac_name+' n='+str(n)+', '+str(instances)+' instances, '+str(samples)+' samples'+100*' ')
		Ws,Xs=getdata(n,depth,scaling,instances,samples)

		print(n)

		for i,W in enumerate(Ws):
			fn='outputs/depth='+str(depth)+' AS/'+ac_name+' n='+str(n)+' scaling='+scaling+'/instance '+str(i)

			if os.path.isfile(fn) and bk.get(fn).size>=samples:
				continue
			print_('instance '+str(i+1),mode)
			Xs_,_=split_data(Xs,'instance')
			instance=GPU_sum.sum_perms_multilayer(W,Xs_,ac_name,mode='silent')
			bk.save(instance,fn)

def generate_zip(nmin,nmax,depth,ac_name,scaling,samples,mode='standard'):

		
	for n in range(nmin,nmax+1):
		print_('zip ',ac_name,' n=',n,', ',samples,' samples',100*' ')
		Ws,Xs=getdata(n,depth,scaling,samples,samples)

		fn=str_('zipoutputs/depth=',depth,' AS/'+ac_name+' n=',n,' '+scaling)

		if os.path.isfile(fn) and bk.get(fn).size>=samples:
			continue

		Ws,Xs=split_zip(Ws,Xs)
		instance=GPU_sum.sum_perms_multilayer_zip(Ws,Xs,ac_name,mode='silent')
		bk.save(instance,fn)



if __name__=='__main__':

	if len(sys.argv)==1:
		print('\n\ngen_outputs.py nmin nmax depth activation X/H (i)nstance/(z)ip instances (samples required if (i)nstance)\n\n')
		quit()
	nmin=int(sys.argv[1])
	nmax=int(sys.argv[2])
	depth=int(sys.argv[3])
	ac_name=sys.argv[4]
	scaling=sys.argv[5]
	combinemode=sys.argv[6]
	instances=int(sys.argv[7])

	if combinemode=='i':
		samples=int(sys.argv[8])
		generate(nmin,nmax,depth,ac_name,scaling,instances,samples)

	if combinemode=='z':
		generate_zip(nmin,nmax,depth,ac_name,scaling,instances)

