import re
from collections import deque
import jax.numpy as jnp
import math


BOX='\u2588'
box='\u2592'
dash='\u2015'
infty='\u221E'

arrowup='\u2191'
arrowdown='\u2193'
arrowleft='\u2190'
arrowright='\u2192'


def indent(s):
    return '\n'.join(['    '+l for l in s.splitlines()])


####


def boxedsidebyside(*elements,separator=''):
    elements=[vsqueeze(e) for e in elements]
    height=max([len(e.splitlines()) for e in elements])
    Es=[makeBOX(e,height=height,border=box) for e in elements]
    if separator!='':
        sepbox=makeBOX(separator,height=height,border=' ')
        Es=('NEXTBOX'+sepbox+'NEXTBOX').join(Es).split('NEXTBOX')
    return '\n'.join([''.join(line) for line in zip(*[E.splitlines() for E in Es])])

def vsqueeze(S):
    lines=S.splitlines()
    while lines[-1]=='': lines.pop()
    return '\n'.join(lines)

def padwidth(S,width,fillstyle=' ',border='',**kw):
    return '\n'.join([border+l+(width-len(l))*fillstyle+border for l in S.splitlines()])

def padheight(S,height,align='center',**kw):
    h=len(S.splitlines()); d=height-h
    match align:
        case 'center': a=d//2
        case 'top': a=0
        case 'bottom': a=d
    b=d-a
    return a*'\n'+S+b*'\n'+' '

def makebox(S,width=None,height=None,**kw):
    if height==None: height=len(S.splitlines())
    if width==None: width=max([len(l) for l in S.splitlines()])
    return padwidth(padheight(S,height,**kw),width,**kw)

def makeBOX(S,width=None,height=None,border=' '):
    return addborder(addborder(makebox(S,width,height),' '),border)


def addborder(S,border):
    lines=S.splitlines()    
    lines=[border*len(lines[0])]+lines+[border*len(lines[0])]
    return '\n'.join([border+border+l+border+border for l in lines])




def sidebyside(*elements,separator=' ',**kw):
    height=max([len(e.splitlines()) for e in elements])
    Es=[makebox(e,height=height,**kw) for e in elements]
    return '\n'.join([separator.join(line) for line in zip(*[E.splitlines() for E in Es])])

def drawtree(tree,parseleaf=None):
    if parseleaf==None: parseleaf=lambda l: '--'+str(l)+'--'
    if isinstance(tree,list) or isinstance(tree,tuple):
        style=BOX if isinstance(tree,list) else box

        branches=[drawtree(branch,parseleaf) for branch in tree]
        tbranches=sidebyside(*branches,align='top')

        bars=tbranches.splitlines()[0]
        return fillspan(midpoints(bars),style=style)+'\n'+midpoints(bars,style=style)+'\n'+tbranches
    else:
        leaf=parseleaf(tree)
        return len(leaf)*BOX+'\n'+leaf


def midpoints(s,style=BOX):
    def midpoint(B):
        if B=='': return ''
        l=len(B)
        a=l//2
        return a*' '+style+(l-a-1)*' '
    return ' '.join([midpoint(B) for B in s.split(' ')])


def fillspan(points,style=BOX):
    l,c,r=re.fullmatch('( *)([^ ].*?)( *)',points).groups()
    return l+len(c)*style+r

######

def draw_weightdims_tree(weights):
    def parsearray(w):
        if w==None: return 'None'
        try: return str(w.shape)
        except: return type(w).__name__
    return drawtree(weights,parseleaf=parsearray)


######


def overwrite(s1,s2):
    return '\n'.join([''.join([a if b==' ' else b for (a,b) in zip(l1+' '*len(l2),l2+' '*len(l1))])\
    for l1,l2 in zip(s1.splitlines(),s2.splitlines())])
    
def layer(*strings):
    out=strings[0]
    for s in strings[1:]: out=overwrite(out,s)
    return out

######

def breakat(s,b):
    return (b+'\n').join(s.split('b'))


def placelabels(positions,labels):

    if type(labels)!=list: labels=len(positions)*[labels]
    positions,labels=[deque(_) for _ in zip(*sorted(zip(positions,labels)))]
    s=''

    for i in range(positions[-1]+1):
        while positions[0]<=i:
            s=s[:i]+str(labels.popleft())
            positions.popleft()
            if len(positions)==0: return s
        s+=' '

    return s

def cleanstring(s):
    return ''.join(c if c.isalpha() or c.isdigit() else '_' for c in s)

def roundspacingandprecision(spacing,levels=None):
    if levels==None: levels=[1,2.5,5,10]
    precision=math.floor(jnp.log10(spacing))
    roundedspacing=max([t for t in [10**precision*s for s in levels] if t<=spacing])
    return roundedspacing,max(0,-precision+1)

def roundrangeandprecision(center,r,nticks):
	spacing,prec=roundspacingandprecision(2*r/nticks)
	return [spacing*(center//spacing+i) for i in range(round(-nticks//2),round(nticks//2))],prec



def startingfrom(s,*starts):
    if len(starts)==0: return s
    else:
        starts=deque(starts)
        start=starts.popleft()
        s=re.search(start+'.*',s,re.DOTALL).group()
        return startingfrom(s,*starts)




def appendright(l,R):
    indent='\n'+' '*len(l)
    return l+indent.join(R.splitlines())+('\n' if len(R.splitlines())>1 else '')



def findblock(text,pattern):
    inblock=False
    block=[]

    #import pdb; pdb.set_trace()
    
    for i,l in enumerate(text.splitlines()):
        if inblock and (l=='' or l[0]==' '): block.append(l)
        if inblock and l!='' and l[0]!=' ': break
        if not inblock: a=i
        if re.search(pattern,l):
            block.append(l)
            inblock=True

    return a,i,'\n'.join(block)







####################################


def treetest():
    t=[[[1],[1,[1]]],[1],[1]]
    print(t)
    print(drawtree(t))

    t=[[1],[1,[1]]]
    print(t)
    print(drawtree(t))


a='Lorem ipsum dolor sit amet, \nconsectetur adipiscing elit,'
b='sed do eiusmod \ntempor incididunt \nut labore et dolore \nmagna aliqua'

lipsum=a+b


linestyles=['b-','r-','b--','r--','b:','r:']
colors=['b','r','m','g','c']


def test():
    a='Lorem ipsum dolor sit amet, '+\
        '\nconsectetur adipiscing elit,'
    b='sed do eiusmod \n'+\
        'tempor incididunt \n    ut labore et dolore \nmagna aliqua'
#    print(a)
#    print(b)
#    print(boxedsidebyside(a,b))
    c=a+b
    print(c)
    print(findblock(c,'incidi'))
    