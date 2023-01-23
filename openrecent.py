import os
import pickle
import matplotlib.pyplot as plt
#import code

folder='temp'
fns=os.listdir(folder)
times=[os.path.getmtime(os.path.join(folder,fn)) for fn in fns]
timesfns=list(zip(times,fns))
timesfns.sort()
path=os.path.join(folder,timesfns[-1][-1])
print(path)

with open(path,'rb') as f:
    data=pickle.load(f)



#code.interact()