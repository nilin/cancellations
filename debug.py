import jax



#mode='Run'
mode='singlefunction'


if __name__=='__main__':
	with jax.disable_jit():
		import pick_and_run as pr
		import testsinglefunction as tf

		match mode:
			case 'Run':
				pr.Run(**pr.profile).run_as_main()
			case 'singlefunction':
				tf.Run(**tf.profile).run_as_main()
