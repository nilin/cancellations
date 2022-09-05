import os
import re
import curses as cs
import pdb
from dashboard import dash




up='\u2191'
down='\u2193'
left='\u2190'
right='\u2192'

def _pickfolder_(stdscr,msg='select folder',condition=None):

	explainpad=cs.newpad(500,500)
	listpad=cs.newpad(500,500)
	matchinfopad=cs.newpad(500,500)

	s=''
	paths=[d+'/' for d,_,files in os.walk('outputs') if 'and' not in d and 'data' not in d and '|' in d]
	if condition!=None: paths=list(filter(condition,paths))
	ls=0

	while True:
		stdscr.refresh()
		ps='(.*)'.join([c if c!='|' else '\|' for c in s])
		p=re.compile(ps)

		out=list(filter(p.search,paths))
		out.sort(reverse=True,key=lambda s:s[-15:])

		ls=max(0,min(len(out)-1,ls))

		explainpad.erase()
		explainpad.addstr(0,0,dash*len(msg)+'\n'+msg+'\n'+dash*len(msg)+'\n\n'+\
			'move with arrow keys:\n{}: up\n{}: down\n{}: fast up\n{}: fast down'.format(up,down,left,right))
		#+'\n\nor type to filter by substring (BACKSPACE to undo)')
		#explainpad.addstr(10,0,'SEARCH FOR SUBSTRING: '+s)

		explainpad.refresh(0,0,1,1,h1-1,w1-1)
		displayoptions(out,ls,listpad,matchinfopad)
		stdscr.refresh()

		c=stdscr.getch()
		if c==10: 
			stdscr.clear()
			stdscr.refresh()
			return out[ls]
		elif c==127: s=s[:-1]
		elif c==259: ls-=1
		elif c==258: ls+=1
		elif c==260: ls-=5
		elif c==261: ls+=5
		elif c==113: quit()
		#else: s=s+chr(c)

def displayoptions(matches,selection,listpad,matchinfopad):
	matchinfopad.erase()
	listpad.erase()
	for i,match in enumerate(matches):
		listpad.addstr(i,2,'{}: {}        -{}'.format(str(i+1).rjust(3),match,runinfo(match)))
	try:
		matchinfopad.addstr(0,0,runinfo(matches[selection]))
		matchinfopad.addstr(2,0,getinfo(matches[selection]))
	except:
		matchinfopad.addstr(0,0,'no folder selected or no info.txt')
	listpad.addch(selection,0,'>')
	listpad.refresh(max(0,selection-h//2),0,1,w1,h-1,w2-1)
	matchinfopad.refresh(0,0,1,w3,h-1,w-1)


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


def _pickfolders_(stdscr,msg=''):
	listpad=cs.newpad(500,500)
	matchinfopad=cs.newpad(500,500)

	instrpad=cs.newpad(500,500)
	msgpad=cs.newpad(500,500)
	instrpad.addstr('press a to add file,\nBACKSPACE to remove file,\nENTER to finish selection')
	msgpad.addstr(msg)

	stdscr.erase()
	stdscr.refresh()

	folders=[]
	ls=0
	while True:
		instrpad.refresh(0,0,1,1,h1-1,w1-1)
		msgpad.refresh(0,0,h1,1,h-1,w1-1)
		displayoptions(folders,ls,listpad,matchinfopad)
		stdscr.refresh()

		c=stdscr.getch()
		if c==113: quit()
		if c==259: ls-=1
		if c==258: ls+=1
		ls=max(0,min(len(folders)-1,ls))

		if c==10: return folders
		if c==127 and len(folders)!=0: del folders[ls]
		if chr(c)=='a': folders.append(_pickfolder_(stdscr,msg))





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
	


def pickfolders(multiple=True,**kw):

	def wrapped(stdscr):
		cs.use_default_colors()
		global h,w,w1,w2,w3,h1,h2
		h=cs.LINES
		w=cs.COLS
		w1=w//4; w3=(2*w)//3; w2=w3-5
		h1=h//3; h2=(2*h)//3

		return _pickfolders_(stdscr,**kw) if multiple else _pickfolder_(stdscr,**kw)
		#return pickfolder(stdscr)

	return cs.wrapper(wrapped)

if __name__=='__main__':

	print(str(pickfolders()))

#	path,(a,b)=commonanc(f1,f2)
#	cfg.outpath=path+a+' and '+b+'/'
#
#	print(cfg.outpath)
#	
#	plotexamples(f1,f2)
