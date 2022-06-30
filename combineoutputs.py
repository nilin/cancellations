import bookkeep as bk
import jax.numpy as jnp
import os
import re
import sys




def assertdisjoint(fs):
	intervals=sorted([[int(a) for a in f.split(' ')] for f in fs])
	points=[]
	for [a,b] in intervals:
		points.append(a)
		points.append(b)


	print(points)
	print(sorted(points))

	assert(points==sorted(points))	



folder=sys.argv[1]
blockfolder=folder+' blocks'
outfolder=folder+' combined'

path=folder+'/depth=2 AS/'
blockpath=blockfolder+'/depth=2 AS/'
outpath=outfolder+'/depth=2 AS/'



for f in os.listdir(path):


	if f.find('.')!=-1:
		continue

	data=bk.get(path+f)

	if f.find('samples')==-1:

		s=data.shape[0]
		bk.save(data,blockpath+f+'/0 '+str(s))
	else:
		out=re.search('(.*?) samples \[(.*?)\]',f)
		pref=out.group(1)
		interval=out.group(2).replace(', ',' ')
		bk.save(data,blockpath+pref+'/'+interval)
		


nmax=int(sys.argv[2])

xh='X'
acs=['HS','ReLU','tanh','exp']
for ac in acs:
	for n in range(2,nmax+1):
		innerfolder=ac+' n='+str(n)+' '+xh

		assertdisjoint(os.listdir(blockpath+innerfolder))

		blocks=[bk.get(blockpath+innerfolder+'/'+f) for f in os.listdir(blockpath+innerfolder)]
		out=jnp.concatenate(blocks)
		bk.save(out,outpath+innerfolder)
		


#
#folder=input('folder: ')
#blockfolder=folder+'\ blocks'
#outfolder=folder+'\ combined'
#
#path=folder+'/depth=2 AS/'
#path_=folder+'/depth=2\ AS/'
#blockpath=blockfolder+'/depth=2\ AS/'
#outpath=outfolder+'/depth=2\ AS/'
#
#
#
#for f in os.listdir(path):
#
#	f_=f.replace(' ','\ ')
#
#
#	if f.find('.')!=-1:
#		continue
#
#	if f.find('samples')==-1:
#
#		s=bk.get(path+f).shape[0]
#
#		os.system('mkdir -p ./'+blockpath+f_)
#		os.system('cp '+path_+f_+' '+blockpath+f_+'/0,\ '+str(s))
#	else:
#		print(f_)
#		out=re.search('(.*?) samples \[(.*?)\]',f)
#		pref=out.group(1)
#		interval=out.group(2)
#
#		print(pref)
#		print(interval)
#
#		
#		os.system('mkdir -p '+blockpath+pref.replace(' ','\ '))
#		os.system('cp '+path_+f_+' '+blockpath+pref.replace(' ','\ ')+'/'+interval.replace(' ','\ '))
#		
#
#
#acs=['HS','ReLU','tanh','exp']
#for ac in acs:
#	path=
