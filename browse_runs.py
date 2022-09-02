import os
import re
import curses as cs
import pdb





up='\u2191'
down='\u2193'
left='\u2190'
right='\u2192'

def pickfolder(stdscr,msg=''):

	#staticinfopad=cs.newpad(100,100)
	infopad=cs.newpad(500,500)
	searchstringpad=cs.newpad(500,500)
	matchpad=cs.newpad(500,500)
	#staticinfopad.addstr(0,0,msg)

	s=''
	paths=[d+'/' for d,_,files in os.walk('outputs') if 'and' not in d and 'data' not in d and '|' in d]
	ls=0

	while True:
		stdscr.refresh()
		ps='(.*)'.join([c if c!='|' else '\|' for c in s])
		p=re.compile(ps)

		out=list(filter(p.search,paths))
		out.sort(reverse=True,key=lambda s:s.split('|')[-1])

		ls=max(0,min(len(out)-1,ls))

		searchstringpad.erase()
#		try:
#			searchstringpad.addstr(0,0,'Please type subtring to search, e.g. {}'.format(out[len(out)//2][-8:-4]))
#		except:
#			searchstringpad.addstr(0,0,'Please type subtring to search')
		searchstringpad.addstr(0,0,'move with arrow keys:\n{}: up\n{}: down\n{}: fast up\n{}: fast down'.format(up,down,left,right)+\
			'\n\nor type to filter by substring (BACKSPACE to undo)')
		searchstringpad.addstr(10,0,'SEARCH FOR SUBSTRING: '+s)

		searchstringpad.refresh(0,0,1,1,h1-1,w1-1)
		#staticinfopad.refresh(0,0,1,w3,h-1,w-1)
		displayoptions(out,ls,matchpad,infopad)
		stdscr.refresh()

		c=stdscr.getch()
		if c==10: return out[ls]
		elif c==127: s=s[:-1]
		elif c==259: ls-=1
		elif c==258: ls+=1
		elif c==260: ls-=5
		elif c==261: ls+=5
		else: s=s+chr(c)

def displayoptions(matches,selection,matchpad,infopad):
	infopad.erase()
	matchpad.erase()
	for i,match in enumerate(matches):
		matchpad.addstr(i,2,'{}: {}        -{}'.format(str(i+1).rjust(3),match,runinfo(match)))
	try:
		infopad.addstr(0,0,runinfo(matches[selection]))
		infopad.addstr(2,0,getinfo(matches[selection]))
	except:
		infopad.addstr(0,0,'no folder selected or no info.txt')
	matchpad.addch(selection,0,'>')
	matchpad.refresh(max(0,selection-h//2),0,1,w1,h-1,w2-1)
	infopad.refresh(0,0,1,w3,h-1,w-1)


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


def menu(stdscr,msg=''):
	matchpad=cs.newpad(500,500)
	infopad=cs.newpad(500,500)

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
		displayoptions(folders,ls,matchpad,infopad)
		stdscr.refresh()

		c=stdscr.getch()
		if c==259: ls-=1
		if c==258: ls+=1
		ls=max(0,min(len(folders)-1,ls))

		if c==10: return folders
		if c==127 and len(folders)!=0: del folders[ls]
		if chr(c)=='a': folders.append(pickfolder(stdscr,msg))





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
	


def pickfolders():

	def wrapped(stdscr):
		cs.use_default_colors()
		global h,w,w1,w2,w3,h1,h2
		h=cs.LINES
		w=cs.COLS
		w1=w//4; w3=(3*w)//4; w2=w3-5
		h1=h//3; h2=(2*h)//3

		return menu(stdscr)
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
