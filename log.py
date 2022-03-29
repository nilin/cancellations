import time

def log(msg,loud=True):

	msg=msg+' | '+time.ctime(time.time())
	with open('log','a') as f:
		f.write('\n'+str(msg))
	if loud:
		print(msg)
