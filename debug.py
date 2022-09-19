from cancellations.utilities import setup
import jax
from cancellations.utilities import sysutil

sysutil.clearscreen()

#rs=input('debug Run or a single function? (r/s) + ENTER')
rs='r'

print(rs)
match rs:
	case 'r': mode='Run'
	case 's': mode='singlefunction'


if __name__=='__main__':
	with jax.disable_jit():
		import pick_and_run as pr

		match mode:
			case 'Run':
				pr.Run(**pr.profile).run_as_main()
			case 'singlefunction':
				import testsinglefunction as tf
