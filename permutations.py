# nilin

import numpy as np
import math
import pickle
import time
import copy
import bookkeep as bk
import util
import jax
import sys
import jax.numpy as jnp
import bisect
import time
	
def nextperm(p_):
	p=list(p_)
	n=len(p)
	i=n-1
	while p[i-1]>p[i]:
		i=i-1
		if i==0: #wrap around
			return list(range(n)),(-1)**(n//2)
		
	first_left_downstep=i-1
	last_upstep=i
	
	while i+1<n and p[i+1]>p[first_left_downstep]:
		i=i+1
	last_above_downstep=i

	p[first_left_downstep],p[last_above_downstep]=p[last_above_downstep],p[first_left_downstep]
	pnext=p[:last_upstep]+list(reversed(p[last_upstep:]))

	paritydiff=(-1)**(((n-last_upstep)//2+1)%2) #parity difference

	return pnext,paritydiff
"""
next permutation,
parity moves by this much. Reversal yields r//2 swaps where reversed part has length r. Plus one previous swap, parity moves by r//2+1.
"""
		

def nextblock(p,k):
	n=len(p)
	sel=perm_to_selections(p[:n-k])
	i=n-k-1
	dsign=1
	while(i>-1 and sel[i]==n-i-1):
		sel[i]=0
		dsign=dsign*(-1)**(n-i-1)
		i=i-1
	if not i==-1:
		sel[i]=sel[i]+1
		dsign=dsign*-1
	return selections_to_perm(sel+k*[0]),dsign
		
		
def descpower(n,k):
	p=1
	for i in range(k):
		p=p*n
		n=n-1
	return p

def generate_seq_blocks(k_large,k_small,n):
	skipfunction=nextperm if k_small==0 else lambda p:nextblock(p,k_small)	
	p=list(range(n))	
	sign=1
	block=[]
	signs=[]
	for i in range(descpower(k_large,k_large-k_small)):
		block.append(p)
		signs.append(sign)
		p,ds=skipfunction(p)	
		sign=sign*ds
	return jnp.array(block),jnp.array(signs)


def gen_complementary_perm_seqs(ks,n=None):
	if n==None:	
		n=ks[0]
	ks_=ks+[0]
	out=[generate_seq_blocks(ks_[i],ks_[i+1],n) for i in range(len(ks))]

	bk.log('blocksizes '+str(ks[0])+'! = '+'*'.join(['('+str(ks_[i])+'!/'+str(ks_[i+1])+'!)' for i in range(len(ks))])+' = '+'*'.join([str(len(b[1])) for b in out]))
	return out


def gen_complementary_Perm_seqs(ks,**kwargs):
	return [(perm_as_matrix(p),signs) for p,signs in gen_complementary_perm_seqs(ks,**kwargs)]
		

def compose(*perms):
	if len(perms)==2:
		p2,p1=perms
		return [p2[p] for p in p1]
	else:
		newlist=perms[:-2]+(compose(perms[-2],perms[-1]),)
		return compose(*newlist)



def perm_to_selections(p):
	n=len(p)	
	seen=[]
	selections=[]
	for i in range(n):
		s=p[i]-sum(1 for v in seen if v<p[i])
		selections.append(s)
		seen.append(p[i])
	return selections


def selections_to_perm(S):
	n=len(S)
	options=list(range(n))
	p=[]
	for i in range(n):
		s=S[i]	
		p.append(options[s])	
		options=options[:s]+options[s+1:]
	return p


def embed(p,indices,n):
	m=len(p)
	p_=list(range(n))
	for k in range(m):
		i=indices[k]
		j=indices[p[k]]
		p_[i]=j
	return p_


def perm_to_k(p):
	selections=perm_to_selections(p)
	n=len(p)	
	base=1
	k=0
	for i in range(1,n+1):
		j=n-i
		k=k+base*selections[j]
		base=base*i
	return k

def k_to_perm(k,n):
	s=[]
	base=1
	r=k
	for base in range(1,n+1):
		s.append(r%base)
		r=r//base
	s.reverse()
	return selections_to_perm(s)


def k_to_matrix(k,n):
	return perm_as_matrix(k_to_perm(k,n))


ReLU=lambda x:(jnp.abs(x)+x)/2

@jax.jit
def perm_as_matrix(p):
	p=jnp.array(p)
	n=p.shape[-1]
	i_minus_pj=jax.vmap(jnp.add,in_axes=(0,None),out_axes=(-2))(jnp.arange(n),-p)
	delta=lambda x:ReLU(-jnp.square(x)+1)
	return delta(i_minus_pj)


def perm_as_matrix2(p_):
	p=jnp.array(p_)
	n=p.shape[-1]
	return jnp.eye(n)[jnp.array(p)].T


"""
Best to update sign using nextperm
"""
@jax.jit
def sign(p):
	n=len(p)
	p_=jnp.array(p)
	pi_minus_pj=p_[:,None]-p_[None,:]
	pi_gtrthan_pj=jnp.heaviside(pi_minus_pj,0)
	inversions=jnp.sum(jnp.triu(pi_gtrthan_pj))
	return 1-2*(inversions%2)


def printperm(p):
	print('k: '+str(perm_to_k(p)))
	print('p: '+str([i+1 for i in p]))
	print('sign: '+str(sign(p)))
	print(perm_as_matrix(p))
	print()
	
def id(n):
	return list(range(n))



"""
tests----------------------------------------------------------------------------------------------------s
"""

	

def performancetest(n):

	N=100
	clock=bk.Stopwatch()

	p=id(n)
	s=1
	for i in range(N):
		p,ds=nextperm(p)
		s=s*ds
	print('next_perm '+str(N/clock.tick())+'/second')


	p=id(n)
	s=1
	for i in range(N):
		p,ds=nextblock(p,4)
		s=s*ds
	print('nextblock '+str(N/clock.tick())+'/second')

	p=id(n)
	s=1
	for i in range(N):
		p,ds=nextperm(p)
		perm_to_selections(p)
		s=s*ds
	print('perm_to_selections '+str(N/clock.tick())+'/second')


	for i in range(N):
		k_to_perm(i,n)
	print('k_to_perm '+str(N/clock.tick())+'/second')

	p=id(n)
	for i in range(N):
		perm_as_matrix(p)
		p,_=nextperm(p)
	print('perm_as_matrix: '+str(N/clock.tick())+'/second')


	p=id(n)
	for i in range(N):
		perm_as_matrix2(p)
		p,_=nextperm(p)
	print('perm_as_matrix2: '+str(N/clock.tick())+'/second')

	p=id(n)
	s=1
	for i in range(N):
		assert(s==sign(p))
		p,ds=nextperm(p)
		s=s*ds
	print('sign: '+str(N/clock.tick())+'/second')


	
def test(n):

	print('generated from k'+100*'-')
	for k in range(2*math.factorial(n)):
		p=k_to_perm(k,n)
		printperm(p)
		verify(k,p)


	p=list(range(n))
	s=1
	print('\nsequentially generated'+100*'-')
	for k in range(2*math.factorial(n)):
		printperm(p)
		verify(k,p)
		p,ds=nextperm(p)
		s=s*ds
		assert(s==sign(p))
		print(str(s)+' '+str(sign(p)))


	p=list(range(n))
	s=1
	print('\nsequentially generated with skip 3!'+100*'-')
	for k in range(0,2*math.factorial(n),6):
		printperm(p)
		verify(k,p)
		p,ds=nextblock(p,3)
		s=s*ds
		assert(s==sign(p))
		print(str(s)+' '+str(sign(p)))


	(P,sp),(Q,sq),(R,sr)=gen_complementary_perm_seqs([5,4,2])
	permnumber=0
	for i in range(sp.size):
		for j in range(sq.size):
			for k in range(sr.size):
				s=sp[i]*sq[j]*sr[k]
				perm=compose(P[i],Q[j],R[k])
				verify(permnumber,perm)
				printperm(perm)
				permnumber=permnumber+1
	print(R)
	print(Q)
	print(P)
	


def verify(k,p):		
	n=len(list(p))
	assert(perm_to_k(p)==k%math.factorial(n))
	assert(k_to_perm(k,n)==p)		
	assert(selections_to_perm(perm_to_selections(p))==p)
	assert(jnp.abs(sign(p)-jnp.linalg.det(perm_as_matrix(p)))<.001)

"""
tests----------------------------------------------------------------------------------------------------s
"""
if __name__=='__main__':
	if len(sys.argv)>1 and sys.argv[1]=='t':
		test(int(input('n for test: ')))
		performancetest(int(input('n for perf test: ')))


