import colorama
colorama.init()
import sys
import os


def gotoline(n):
	print('\x1b['+str(n)+';0H')

def clear():
	os.system('cls' if os.name == 'nt' else 'clear')

def overwrite(n,k):
	gotoline(n)
	for i in range(k):
		print()
	gotoline(n)

#----------------------------------------------------------------------------------------------------

BOX='\u2588'
box='\u2592'





if __name__=='__main__':
	clear()
