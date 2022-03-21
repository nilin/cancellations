import numpy as np
import math
import pickle
import time
import bookkeep as bk
import copy
import jax
import jax.numpy as jnp
import util
import sys
import os
import shutil
import multiprocessing as mp
import permutations
import partialsum as ps






"""
gen_partial_sum.py ReLU n 12 40320 0
"""



ac_name=sys.argv[1]
Wtype=util.Wtypes[sys.argv[2]]
n=int(sys.argv[3])
start=int(sys.argv[4])
stop=int(sys.argv[5])

dirpath='partialsums/'+Wtype
bk.mkdir('data/'+dirpath)

W,X=[bk.getdata(Wtype+'/WX')[k][n] for k in ('Ws','Xs')]

print('Computing partial sum for '+ac_name+' activation, '+Wtype+' weights, and n='+str(n)+'.')
prefix=dirpath+'/'+ac_name+' n='+str(n)+' range='


S=ps.partial_sum((W,X,ac_name,start,stop-start))

filepath=prefix+str(start)+' '+str(stop)			
bk.savedata({'result':S,'interval':(start,stop),'W':W,'X':X},filepath)
