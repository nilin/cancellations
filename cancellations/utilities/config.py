from . import tracking,sysutil
import sys
import time
from collections import deque



biasinitsize=.1
plotfineness=50

def agrees(d1,**d2):
    return all([d1[k]==d2[k] for k in d1.keys() if k in d2.keys()])


breaker=tracking.Breaker()




def test():
    print(agrees(dict(a=1,b=2,c=3),b=1))
    print(agrees(dict(a=1,b=2,c=3),b=2))