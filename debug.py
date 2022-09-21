from cancellations.utilities import setup, tracking, browse
from cancellations.display import _display_
import jax
from cancellations.utilities import sysutil

sysutil.clearscreen()

#rs=input('debug Run or a single function? (r/s) + ENTER')
#rs='r'
#
#print(rs)
#match rs:
#	case 'r': mode='Run'
#	case 's': mode='singlefunction'
#


def debug(mode):
	with jax.disable_jit():
		import pick_and_run as pr

		match mode:
			case 'Run':
				pr.Run(pr.profile).run_as_main()
			case 'singlefunction':
				import testsinglefunction as tf



def getmode():

	profile=browse.Browse.getdefaultprofile().butwith(options=['Run','singlefunction'])
	process=browse.Browse(profile)
	return process.run_as_main()


if __name__=='__main__':
	debug(getmode())