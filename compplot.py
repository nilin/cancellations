from re import I
from browse import pickfolders,commonanc
import browse
import os
import shutil
import util
import config as cfg
import re
import curses as cs
import pdb
import jax.numpy as jnp
import exampletemplate
import matplotlib.pyplot as plt
from config import session
from display import clear


def process_snapshot(processed,f,weights,X,Y,i):
	processed.addcontext('minibatchnumber',i)
	processed.remember('Af norm',jnp.average(f(weights,X[:100])**2))
	processed.remember('test loss',util.SI_loss(f(weights,X),Y))
	processed.remember('weight norms',util.applyonleaves(weights,util.norm))

def plotexamples(paths):

	processedruns=[]
	learners=[]

	for i_run,path in enumerate(paths):

		setup=cfg.load(path+'data/setup')
		unprocessed=cfg.load(path+'data/unprocessed')

		learner,X,Y=setup['learner'],setup['X_test'],setup['Y_test']
		pfunc=learner.restore().getemptyclone()

		weightslist,i_s=unprocessed.gethist('weights','minibatchnumber')

		processed=cfg.Memory()
		print('processing run {}'.format(i_run+1))
		for imgnum,(weights,i) in enumerate(zip(weightslist,i_s)):
			print('image {} of {}'.format(imgnum+1,len(i_s)),end='\r')

			cfg.trackcurrenttask('processing snapshots for learning plot',(imgnum+1)/len(weightslist))
			process_snapshot(processed,pfunc.f,weights,X,Y,i)		

		processedruns.append(processed)
		learners.append(learner)
		print()

		

	plt.close('all')
	fig,(ax0,ax1)=plt.subplots(1,2,figsize=(15,7))
	#fig.suptitle('test loss '+)

	lstyles=['r-','b--','g:','m:','ro-','bo--','go:','mo:']

	for ls,processed,learner in zip(lstyles,processedruns,learners):
		ax0.plot(*util.swap(*processed.gethist('test loss','minibatchnumber')),ls,label=learner.richtypename())
		ax1.plot(*util.swap(*processed.gethist('test loss','minibatchnumber')),ls,label=learner.richtypename())
	ax0.legend()
	ax0.set_xscale('log')
	ax0.set_ylim(bottom=0,top=1.1)
	ax0.grid(True,which='major',ls='-',axis='y')
	ax0.grid(True,which='minor',ls=':',axis='y')

	ax1.legend()
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	ax1.grid(True,which='major',ls='-',axis='y')
	ax1.grid(True,which='minor',ls=':',axis='y')

	print(cfg.outpath)

	cfg.savefig('{}{}'.format(cfg.outpath,'losses.pdf'),fig=fig)

#	try:
#		fig,ax=plt.subplots(1)
#		weights1,I1=processed[1].gethist('weight norms','minibatchnumber')
#		weights2,I2=processed[2].gethist('weight norms','minibatchnumber')
#		ax.plot(I1,[util.recurseonleaves(ws,max) for ws in weights1],'b-',label=learners[1].richtypename())
#		ax.plot(I2,[util.recurseonleaves(ws,max) for ws in weights2],'r--',label=learners[2].richtypename())
#		ax.legend()
#		cfg.savefig('{}{}'.format(cfg.outpath,'weights.pdf'),fig=fig)
#	except:
#		print('weights plot failed')
		


#cfg.print_task_on_poke()










_,folders=browse.pickfolders_leave_cs()
outpath,branches=commonanc(*folders)


clear()
cfg.outpath=outpath+' and '.join([b[:-1] for b in branches])+'/'
if os.path.exists(cfg.outpath):
	for i in range(100):
		path_i=cfg.outpath[:-1]+' ({})/'.format(i)
		if not os.path.exists(path_i):
			cfg.outpath=path_i
			break

os.makedirs(cfg.outpath,exist_ok=True)	
for folder,branch in zip(folders,branches):
	shutil.copytree(folder,cfg.outpath+branch)
#print(folders)
#print(outpath)
#print(branches)
#

innernames=[]
for folder in folders:
	for fn in os.listdir(folder):
		try:
			int(fn[0])
			if fn not in innernames:
				print('appending '+fn)
				innernames.append(fn)
		except:
			print('discarding '+fn)
			break

for fn in innernames:
	nfiles=sum([os.path.exists(folder+fn) for folder in folders])
	for folder,branch in zip(folders,branches):
		print(fn)
		newfolder=cfg.outpath+fn[:-4]+' ({} runs)'.format(nfiles)+'/'
		newpath=newfolder+branch[:-1]+'.pdf'
		os.makedirs(newfolder,exist_ok=True)	
		try: shutil.copy(folder+fn,newpath)
		except: pass





plotexamples(folders)
	
	#plotexamples(folders)
