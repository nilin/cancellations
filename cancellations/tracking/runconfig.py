import os
from cancellations.tracking import sysutil
import json

#import sys
#import jax
#display_on=True
#debug=False
#biasinitsize=.1

initweight_coefficient=2

if not os.path.exists('config/config.json'):
    sysutil.write('{}','config/config.json')
        
f=open('config/config.json')
cfgdata=json.load(f)