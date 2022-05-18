import jax
import jax.numpy as jnp
import GPU_sum
import bookkeep as bk
import math
import testing
import util
import sys
import os


def split_data(Xs):
	samples,n,d=Xs.shape
	batchsize=max(round(10000/math.factorial(n)),1)
	start=0
	batches=[]
	while start<samples:
		end=start+min(batchsize,samples)
		batches.append(Xs[start:end])
		start=end
	return batches




def getdata(n,depth,scaling,instances,samples):
	Ws=bk.get('inputs/Ws/n='+str(n)+' depth='+str(depth)+' scaling='+scaling)
	Xs=bk.get('inputs/Xs/n='+str(n))
	
	Ws=Ws[:min(len(Ws),instances)]
	Xs=Xs[:min(Xs.shape[0],samples)]

	return Ws,Xs

def inspect_data(Ws,Xs,ac_name,mode):
	print_('\n'+100*'='+'\ndata spec '+ac_name+' n='+str(Xs.shape[-2]),mode)
	print_('W: '+str(len(Ws))+' instances of '+str([W.shape for W in Ws[0]])+' (depth='+str(depth)+')',mode)
	print_('X: '+str(Xs.shape[0])+' samples of '+str(Xs.shape[1:])+'\n'+100*'='+'\n',mode)


def print_(s,mode,**kwargs):
	if mode!='silent':
		print(s,**kwargs)


"""
"""


def generate(nmin,nmax,depth,ac_name,scaling,instances,samples,mode='standard'):

	NN_nd=testing.get_NN_nd(ac_name)
	for n in range(2,25):
		fn='outputs/depth='+str(depth)+' NS/'+ac_name+' n='+str(n)+' scaling='+scaling
		if os.path.isfile(fn):
			continue			
		Ws,Xs=getdata(n,depth,scaling,100,1000)
		NS=jnp.stack([NN_nd(W,Xs) for W in Ws],axis=0)
		bk.save(NS,fn)
		print_('nonsymmetrized n='+str(n),mode,end='\r')
	for n in range(nmin,nmax+1):
		print(ac_name+' n='+str(n)+', '+str(instances)+' instances, '+str(samples)+' samples'+100*' ')
		Ws,Xs=getdata(n,depth,scaling,instances,samples)
		#inspect_data(Ws,Xs,ac_name,mode)
		for i,W in enumerate(Ws):
			fn='outputs/depth='+str(depth)+' AS/'+ac_name+' n='+str(n)+' scaling='+scaling+'/instance '+str(i)

			if os.path.isfile(fn) and bk.get(fn).size>=samples:
				continue
			print_('instance '+str(i+1),mode)
			Xs_=split_data(Xs)
			instance=GPU_sum.sum_perms_multilayer(W,Xs_,ac_name,mode='silent')
			#instance=GPU_sum.sum_perms_multilayer(W,Xs_,ac_name,mode=('standard' if n>9 else 'silent'))
			bk.save(instance,fn)




if __name__=='__main__':

	if sys.argv[1]=='h':
		print('\n\ngen_outputs.py nmin nmax depth activation=ReLU scaling=X/H instances=10 samples=100\n\n')
		quit()
	nmin=int(sys.argv[1])
	nmax=int(sys.argv[2])
	depth=int(sys.argv[3])
	ac_name=sys.argv[4]
	scaling=sys.argv[5]
	instances=int(sys.argv[6])
	samples=int(sys.argv[7])
	mode=(sys.argv[8] if len(sys.argv)>8 else 'standard')

	gen_outputs.generate(nmin,nmax,depth,ac_name,scaling,instances,samples,mode)
