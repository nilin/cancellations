import re
from collections import deque
import jax.numpy as jnp
import math


BOX='\u2588'
box='\u2592'
dash='\u2015'
infty='\u221E'



def indent(s):
    return '\n'.join(['    '+l for l in s.splitlines()])


####


def sidebyside(*elements,separator=''):
    height=max([len(e.splitlines()) for e in elements])
    Es=[makebox(e,height=height,border=box) for e in elements]
    if separator!='':
        sepbox=makebox(separator,height=height,border=' ')
        Es=('NEXTBOX'+sepbox+'NEXTBOX').join(Es).split('NEXTBOX')
    return '\n'.join([''.join(line) for line in zip(*[E.splitlines() for E in Es])])

def padwidth(S,width,fillstyle=' ',border=''):
    return '\n'.join([border+l+(width-len(l))*fillstyle+border for l in S.splitlines()])

def padheight(S,height):
    h=len(S.splitlines()); d=height-h
    a=d//2; b=d-a;
    return a*'\n'+S+b*'\n'

def makebox(S,width=None,height=None,border=' '):
    if height==None: height=len(S.splitlines())
    if width==None: width=max([len(l) for l in S.splitlines()])
    return addborder(addborder(padwidth(padheight(S,height),width),' '),border)

def addborder(S,border):
    lines=S.splitlines()    
    lines=[border*len(lines[0])]+lines+[border*len(lines[0])]+[border*len(lines[0])]
    return '\n'.join([border+l+border for l in lines])

    
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



def roundspacingandprecision(spacing,levels=None):
    if levels==None: levels=[1,2.5,5,10]
    precision=math.floor(jnp.log10(spacing))
    roundedspacing=max([t for t in [10**precision*s for s in levels] if t<=spacing])
    return roundedspacing,max(0,-precision+1)

def roundrangeandprecision(center,r,nticks):
	spacing,prec=roundspacingandprecision(2*r/nticks)
	return [spacing*(center//spacing+i) for i in range(round(-nticks//2),round(nticks//2))],prec



def startingfrom(s,start):
    return re.search(start+'.*',s).group()




def test():
    a='Lorem ipsum dolor sit amet, \nconsectetur adipiscing elit,'
    b='sed do eiusmod \ntempor incididunt \nut labore et dolore \nmagna aliqua'
    print(a)
    print(b)
    print(sidebyside(a,b))
    