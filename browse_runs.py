import os
import re
import curses as cs
import pdb
from display import dash
import config as cfg
import cdisplay
import time


up='\u2191'
down='\u2193'
left='\u2190'
right='\u2192'

def _pickfolders_(msg='select folder',condition=None):

	W=cfg.currentprofile().dashboard.width
	H=cfg.currentprofile().dashboard.height
	stdscr=cfg.screen
	stdscr.nodelay(False)
	explainpad=cs.newpad(500,500)
	listpad=cs.newpad(500,500)
	matchinfopad=cs.newpad(500,500)

	explanation=\
		cfg.wraptext(msg)+'\n\n'\
		+'Move with arrow keys:\n{}: up\n{}: down\n{}: fast up\n{}: fast down'.format(up,down,left,right)\
		+'\n\nPress SPACE or a to add (i.e. mark) elements. \nPress ENTER to finish selection'
	explainpad.addstr(0,0,explanation)
	explainpad.refresh(0,0,3,5,H-3,W//3)

	def displayoptions(options,selection,selections,listpad,matchinfopad):
		matchinfopad.erase()
		listpad.erase()
		for i,match in enumerate(options):
			listpad.addstr(i,3,'{}: {}        -{}'.format(str(i+1).rjust(3),match,runinfo(match)))
		try:
			matchinfopad.addstr(0,0,runinfo(options[selection]))
			matchinfopad.addstr(2,0,getinfo(options[selection]))
		except:
			matchinfopad.addstr(0,0,'no folder selected or no info.txt')
		
		listpad.addch(selection,0,'>')
		for s in selections: listpad.addch(s,1,'*')
		listpad.refresh(max(0,selection-H//2),0,  3,W//3,  H-3,2*W//3)
		matchinfopad.refresh(0,0,  3,2*W//3,  H-3,W-5)

	paths=[d+'/' for d,_,files in os.walk('outputs') if 'and' not in d and 'data' not in d and '|' in d]
	if condition!=None: paths=list(filter(condition,paths))
	ls=0
	choices=[]

	paths.sort(reverse=True,key=lambda s:s[-15:])
	while True:

		ls=max(0,min(len(paths)-1,ls))
		displayoptions(paths,ls,choices,listpad,matchinfopad)

		c=stdscr.getch()
		if c==97 or c==32: choices.append(ls)
		elif c==127 and ls in choices: choices.remove(ls)
		elif c==259: ls-=1
		elif c==258: ls+=1
		elif c==260: ls-=5
		elif c==261: ls+=5
		elif c==115:
			try: ls=max([c for c in choices if c<ls])
			except: pass
		elif c==99:
			try: ls=min([c for c in choices if c>ls])
			except: pass
		elif c==10: break
		elif c==113: quit()

	stdscr.nodelay(True)
	return paths[ls],[paths[ls] for ls in choices]



def runinfo(folder):
	try:
		with open(folder+'duration','r') as f:
			return 'duration {} s'.format(f.readline())
	except:
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


browsingprofile=cfg.Profile()


def pickfolders(**kw):
	return cdisplay.subtask_in_display(_pickfolders_,browsingprofile,**kw)


#
if __name__=='__main__':
	import cdisplay
	cdisplay.run_in_display(_pickfolders_,cfg.Profile())
