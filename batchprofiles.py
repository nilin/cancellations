import exampletemplate
import example
import config
import sys
import itertools


ACS=[[]]
for i in range(3):
	ACS=[acs+[ac] for acs in ACS for ac in ['tanh','lrelu']]
for acs in ACS:
	print(acs)


i=int(sys.argv[1])
its=int(sys.argv[2])

config.loadtargetpath='outputs/example/09-07|10h56m10s/'
#config.loadtargetpath='outputs/example/09-07|01h38m59s/'
config.learnerparams['activations']=ACS[i]
config.trainingparams['iterations']=its

exampletemplate.runexample(example.prep_and_run)

