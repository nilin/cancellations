import os
import pickle
import matplotlib.pyplot as plt
from cancellations.tracking import browse
#import code

folder='batchoutputs'

while os.path.isdir(folder):
    fns=os.listdir(folder)
    times=[os.path.getmtime(os.path.join(folder,fn)) for fn in fns]
    timesfns=list(sorted(zip(times,fns),reverse=True))
    paths=[os.path.join(folder,fn) for t,fn in timesfns if fn[0]!='.']
    folder=browse.Browse(options=paths).browse_nodisplay()

path=folder

with open(path,'rb') as f:
    data=pickle.load(f)



#code.interact()