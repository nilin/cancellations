import jax



mode='Run'
#mode='singlefunction'


if __name__=='__main__':
	with jax.disable_jit():
		import pick_and_run as pr

		match mode:
			case 'Run':
				pr.Run(**pr.profile).run_as_main()
			case 'singlefunction':
				import testsinglefunction as tf
