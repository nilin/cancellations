#
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
import scratchwork as sw


datafolder=util.Wtypes[input('W-type: n,s,ns,ss')]

Ws=bk.getdata(datafolder+'/WX')['Ws']
Ws_ordered={k:sw.greedycloseordering(W) for k,W in Ws.items() if k>=2 and k<=12}

bk.savedata(Ws_ordered,datafolder+'/forplots/W_ordered')
