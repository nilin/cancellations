import numpy as np
import math
import pickle
import time
import copy
import jax
import jax.numpy as jnp
import os




class Stopwatch:
	def __init__(self):
		self.time=0
		self.tick()

	def tick(self):
		elapsed=time.perf_counter()-self.time
		self.time=time.perf_counter()
		return elapsed


def mkdir(path):
	try:
		os.mkdir(path)
	except OSError:
		pass

def savedata(data,filename):
        filename='data/'+filename
        with open(filename,'wb') as file:
                pickle.dump(data,file)

def getdata(filename):
	with open('data/'+filename,"rb") as file:
		data=pickle.load(file)
	return data

def rangevals(_dict_):
	range_vals=jnp.array([[k,v] for k,v in _dict_.items()]).T
	return range_vals[0],range_vals[1]


def getplotdata(filename):
	data=getdata(filename)
	if type(data) is dict:
		return rangevals(data)
	else:
		return data[0],jnp.array(data[1])


def plot_dict(_dict_,connect,scatter,color='r'):
	_range,vals=rangevals(_dict_)
	print('['+str(jnp.min(vals))+','+str(jnp.max(vals))+']')
	if scatter:
		plt.scatter(_range,vals,color=color)
	if connect:
		plt.plot(_range,vals,color=color)


def saveplot(datanames,savename,colors,moreplots=[],draw=False,connect=False,scatter=True):
	plt.figure()
	plt.yscale('log')
	for i in range(len(datanames)):
		filename=datanames[i]
		color=colors[i]
		data=getdata(filename)
		plot_dict(data,connect,scatter,color=color)
	for _range,vals,color in moreplots:
		plt.plot(_range,vals,color)
	plt.savefig('plots/'+savename+'.pdf')
	if draw:
		plt.show()
			
def progressbar(relwidth,fullwidth):
	return '['+(int(math.ceil(fullwidth*min(relwidth,1))))*'\u2588'+(fullwidth-int(math.ceil(relwidth*fullwidth)))*'_'+']'

def printbar(val,msg=''):
	try:
		printbars([(val,str(msg))])
	except:
		pass
	
def printbars(data):
	print('| '.join([progressbar(val,150//len(data))+'='+str(msg) for val,msg in data]),end='\r')

def printemph(msg):
	print(100*'-'+'\n'+str(msg)+'\n'+100*'-'+'\n')
