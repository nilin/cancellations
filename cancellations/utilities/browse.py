import os
import re
import curses as cs
import pdb

from . import tracking
from ..display.display import dash
from ..display import display as disp
from ..display import cdisplay
from ..utilities import config as cfg
import time


up='\u2191'
down='\u2193'
left='\u2190'
right='\u2192'



def getdefaultprofile():
	profile=tracking.Profile(name='browsing')
	profile.parentfolder='outputs'
	profile.msg='select folder'
	profile.onlyone=False
	profile.regex='(./)?outputs/.*'
	profile.condition1=lambda path:os.path.exists(path+'/data/setup')
	return profile


def _pickfolders_(profile,display):

	browsing=tracking.Process(profile,display=display)

	W=display.width
	H=display.height
	screen=cfg.screen
	screen.nodelay(False)
	explainpad=cdisplay.Pad((0,round(.2*W)),(0,H))
	listpad=cdisplay.Pad((round(.2*W),round(.6*W)),(0,H),100,1000)
	matchinfopad=cdisplay.Pad((round(.7*W),W),(0,H))

	explanation=\
		disp.wraptext(profile.msg)+'\n\n'\
		+'Move with arrow keys:\n{}: up\n{}: down\n{}: fast up\n{}: fast down'.format(up,down,left,right)\
		+('' if profile.onlyone else '\n\nPress SPACE or a to add (i.e. mark) elements.'\
		+'\nPress s or c to move between marked elements.')\
		+'\n\nPress ENTER to finish selection'
	explainpad.addstr(0,0,explanation)
	explainpad.draw()
	screen.refresh()
	


	
	#if profile.matchtype=='dir': paths=[d+'/' for d,_,files in os.walk(profile.parentfolder)]
	#else: paths=['{}/{}/{}'.format(r,d,f) for r,D,F in os.walk(profile.parentfolder) for d in D for f in F]
	#['{}/{}{}'.format(r,'' if len(_d_)==0 else _d_[0]+'/',f) for r,_d_,F in os.walk(profile.parentfolder) for f in F+['']]
	paths=getpaths(profile.parentfolder)
	pattern=re.compile(profile.regex)
	paths=list(filter(pattern.fullmatch,paths))
	paths=list(filter(combineconditions(profile),paths))
	ls=0
	
	filterword=''	# onlyone case
	choices=[]		# multiple case

	paths.sort(reverse=True,key=lambda s:s[-15:])
	while True:

		explainpad.draw()
		ls=max(0,min(len(paths)-1,ls))
		displayoptions(paths,ls,choices,listpad,matchinfopad,profile,H)

		c=cdisplay.extractkey_cs(screen.getch())
		if c=='SPACE' and not profile.onlyone: choices.append(ls)
		elif c=='BACKSPACE':
			if profile.onlyone and len(filterword)>0: filterword=filterword[:-1]
			if not profile.onlyone and ls in choices: choices.remove(ls)
		elif c==259: ls-=1
		elif c==258: ls+=1
		elif c==260: ls-=5
		elif c==261: ls+=5
		elif c=='s':
			try: ls=max([c for c in choices if c<ls])
			except: pass
		elif c=='c':
			try: ls=min([c for c in choices if c>ls])
			except: pass
		elif c=='ENTER': break
		elif c=='q': quit()

	screen.nodelay(True)
	return paths[ls] if profile.onlyone else [paths[ls] for ls in choices]



def displayoptions(options,selection,selections,listpad,matchinfopad,profile,H):
	matchinfopad.erase()
	listpad.erase()
	for i,match in enumerate(options):
		listpad.addstr(i,2,'{}: {}{}'.format(str(i+1),match,getmetadata(match)))
	try:
		matchinfopad.addstr(0,0,getmetadata(options[selection]))
		matchinfopad.addstr(2,0,getinfo(options[selection]))
	except:
		matchinfopad.addstr(0,0,'no folder selected or no info.txt')
	
	listpad.addstr(selection,0,' *' if profile.onlyone else '>')
	for s in selections: listpad.addstr(s,1,'*')
	listpad.refresh(max(0,selection-H//2))
	matchinfopad.draw()



def combineconditions(profile):
	conditions=[]
	for i in range(1,10):
		cname='condition'+str(i)
		if cname in profile.keys(): conditions.append(profile[cname])
		else: break
	def CONDITION(d):
		try: return all([True if c==None else c(d) for c in conditions])
		except: False
	return CONDITION 	

def getpaths(root):
	scan=os.scandir(root)
	out=[]
	for d in scan:
		if d.is_file(): out.append(root+'/'+d.name)
		if d.is_dir():
			out.append(root+'/'+d.name+'/')
			out+=getpaths(root+'/'+d.name)
	return out

def getmetadata(folder):
	try:
		with open(folder+'metadata.txt','r') as f:
			return ' - '+f.readline()
	except Exception as e:
		return ''

def getinfo(path):
	try:
		with open(path+'info.txt','r') as info:
			infostr=''.join(info.readlines())
	except:
		infostr='no info.txt'
	return infostr

def commonanc(*fs):
	levels=list(zip(*[f.split('/') for f in fs]))
	
	path=''
	difflevel=[]
	for l in levels:
		if all([li==l[0] for li in l]):
			path+=l[0]+'/'
		else:
			break
	return path,[f[len(path):] for f in fs]



browsingprofile=tracking.Profile()

def pickfolders(**kw):
	return cdisplay.subtask_in_display(_pickfolders_,browsingprofile,**kw)

def pickfolders_leave_cs(**kw):
	cdisplay.run_in_display(_pickfolders_,browsingprofile,nodelay=False,**kw)
	return browsingprofile.loadedpathandpaths



#
if __name__=='__main__':
	import cdisplay
	cdisplay.session_in_display(_pickfolders_,getdefaultprofile())
