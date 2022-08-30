import os
import util
import config as cfg
import re
import curses as cs
import pdb
import jax.numpy as jnp
import examples
import matplotlib.pyplot as plt
from config import session

rd=0

def pickfolder():

	def f(stdscr):
		cs.use_default_colors()
		h=cs.LINES
		w=cs.COLS

		pad0=cs.newpad(10,w-1)
		pad1=cs.newpad(1000,w-1)
		pad2=cs.newpad(1000,w-1)

		s=''
		paths=[d for d,_,files in os.walk('outputs') if 'and' not in d and 'data' not in d]

		ps='(.*)'

		while True:

			stdscr.refresh()
			p=re.compile(ps)

			out=list(filter(p.search,paths))
			out.sort(reverse=True,key=lambda s:s.split('|')[-1])


			pad0.erase()
			pad1.erase()
			pad2.erase()


			pad0.addstr(0,0,'enter substring of data path. Press esc (twice) to exit')
			pad0.addstr(1,0,s+10*' ')
			pad0.addstr(2,0,ps)
			pad0.addstr(3,0,str(len(out))+' matches')

			for i,d in enumerate(out):
				pad1.addstr(i,0,d)


			if len(out)==1:
				[path]=out
				path+='/'
				with open(path+'info.txt','rb') as info:
					for i,l in enumerate(info.readlines()):
						pad2.addstr(i,0,l)

			x0=rd*w//2+1
			x1=(rd+1)*w//2-1

			pad0.refresh(0,0,1,x0,5,x1)
			pad1.refresh(0,0,5,x0,h//2,x1)
			pad2.refresh(0,0,h//2,x0,h-1,x1)
			stdscr.refresh()
		

			c=stdscr.getch()
			if c==27:
				quit()
			if c==10 and len(out)==1:
				return out[0]+'/'
			if c==127:
				s=s[:-1]
			else:
				s=s+chr(c)

			ps='(.*)'.join([c if c!='|' else '\|' for c in s])


	return cs.wrapper(f)

def commonanc(*fs):
	levels=list(zip(*[f.split('/') for f in fs]))
	
	path=''
	difflevel=[]
	for l in levels:
		if all([li==l[0] for li in l]):
			path+=l[0]+'/'
		else:
			difflevel=l
			break
	return path,difflevel
	
		

def process_snapshot(processed,f,weights,X,Y,i):
	processed.addcontext('minibatchnumber',i)
	processed.remember('Af norm',jnp.average(f(X[:100])**2))
	processed.remember('test loss',util.SI_loss(f(X),Y))
	processed.remember('weight norms',util.applyonleaves(weights,util.norm))

def plotexamples(path1,path2):

	processed=dict()
	learners=dict()

	for x in [1,2]:


		path=locals()['path'+str(x)]
		setup=cfg.load(path+'data/setup')
		unprocessed=cfg.load(path+'data/unprocessed')
		processed[x]=cfg.ActiveMemory()

		learner,X,Y=setup['learner'],setup['X_test'],setup['Y_test']
		learners[x]=learner
		pfunc=learner.restore().getemptyclone()

		weightslist,i_s=unprocessed.gethist('weights','minibatchnumber')

		cfg.logcurrenttask('processing snapshots')
		for imgnum,(weights,i) in enumerate(zip(weightslist,i_s)):
			cfg.trackcurrenttask('processing snapshots for learning plot',(imgnum+1)/len(weightslist))
			process_snapshot(processed[x],pfunc.fwithparams(weights),weights,X,Y,i)		

		

	plt.close('all')

	fig,(ax0,ax1)=plt.subplots(1,2,figsize=(15,7))
	#fig.suptitle('test loss '+)

	ax0.plot(*util.swap(*processed[1].gethist('test loss','minibatchnumber')),'b-',label=learners[1].richtypename())
	ax0.plot(*util.swap(*processed[2].gethist('test loss','minibatchnumber')),'r--',label=learners[2].richtypename())
	ax0.legend()
	ax0.set_ylim(bottom=0,top=1)
	ax0.grid(True,which='major',ls='-',axis='y')
	ax0.grid(True,which='minor',ls=':',axis='y')

	ax1.plot(*util.swap(*processed[1].gethist('test loss','minibatchnumber')),'b-',label=learners[1].richtypename())
	ax1.plot(*util.swap(*processed[2].gethist('test loss','minibatchnumber')),'r--',label=learners[2].richtypename())
	ax1.legend()
	ax1.set_yscale('log')
	ax1.grid(True,which='major',ls='-',axis='y')
	ax1.grid(True,which='minor',ls=':',axis='y')
	cfg.savefig('{}{}'.format(cfg.outpath,'losses.pdf'),fig=fig)

	fig,ax=plt.subplots(1)
	weights1,I1=processed[1].gethist('weight norms','minibatchnumber')
	weights2,I2=processed[2].gethist('weight norms','minibatchnumber')
	ax.plot(I1,[util.recurseonleaves(ws[-1],max) for ws in weights1],'b-',label=learners[1].richtypename())
	ax.plot(I2,[util.recurseonleaves(ws[-1],max) for ws in weights2],'r--',label=learners[2].richtypename())
	ax.legend()
	cfg.savefig('{}{}'.format(cfg.outpath,'weights.pdf'),fig=fig)

def poke(*args,**kw):

	#print('test')
	#print(args)

	try:
		print('{}: {:.0%}'.format(session.getcurrentval('currenttask'),session.getcurrentval('currenttaskcompleteness'))+25*' ',end='\r')
		#print('{:.0%}'.format(session.getcurrentval('currenttaskcompleteness'))+25*' ',end='\r')
	except:
		pass

cfg.poke=poke


if __name__=='__main__':

	rd=0
	f1=pickfolder()
	rd=1
	f2=pickfolder()

	
	print(f1)
	print(f2)

	path,(a,b)=commonanc(f1,f2)
	cfg.outpath=path+a+' and '+b+'/'

	print(cfg.outpath)
	
	plotexamples(f1,f2)
