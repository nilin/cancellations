# nilin

import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import matplotlib
import seaborn as sns
import pickle
import time
import copy
import bookkeep as bk
import util
import jax
import sys
import jax.numpy as jnp
import optax
import bisect
import time
	
def nextperm(p):
	n=len(p)
	i=n-1
	while p[i-1]>p[i]:
		i=i-1
		if i==0: #wrap around
			return list(range(n)),sign(p)
		
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
		

def perm_to_selections(p):
	n=len(p)	
	seen=[]
	selections=[]

	for i in range(n):
		s=p[i]-np.searchsorted(seen,p[i])
		selections.append(s)
		bisect.insort(seen,p[i]) #O(n) :|
	
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


@jax.jit
def perm_as_matrix(p):
	n=len(p)
	i_minus_pj=jnp.arange(n)[:,None]-jnp.array(p)[None,:]
	delta=lambda x:util.ReLU(-jnp.square(x)+1)
	return delta(i_minus_pj)


"""
Best to update sign using nextperm (400 times faster)
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

	for i in range(N):
		k_to_perm(i,n)
	print('k_to_perm '+str(N/clock.tick())+'/second')

	p=id(n)
	for i in range(N):
		perm_as_matrix(p)
		p,_=nextperm(p)
	print('perm_as_matrix: '+str(N/clock.tick())+'/second')


	p=id(n)
	s=1
	for i in range(N):
		assert(s==sign(p))
		p,ds=nextperm(p)
		s=s*ds
	print('sign: '+str(N/clock.tick())+'/second')


	
def test(n):
	p=list(range(n))
	s=1

	print('\nsequentially generated'+100*'-')
	for k in range(2*math.factorial(n)):
		printperm(p)
		verify(k,p,n)
		p,ds=nextperm(p)
		s=s*ds
		assert(s==sign(p))
		print(str(s)+' '+str(sign(p)))

	print('generated from k'+100*'-')
	for k in range(2*math.factorial(n)):
		p=k_to_perm(k,n)
		printperm(p)
		verify(k,p,n)

def verify(k,p,n):		
	assert(perm_to_k(p)==k%math.factorial(n))
	assert(k_to_perm(k,n)==p)		
	assert(selections_to_perm(perm_to_selections(p))==p)
	assert(jnp.abs(sign(p)-jnp.linalg.det(perm_as_matrix(p)))<.001)

"""
tests----------------------------------------------------------------------------------------------------s
"""

if len(sys.argv)>1 and sys.argv[1]=='test':
	test(4)
	performancetest(int(sys.argv[2]))
