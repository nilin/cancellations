import jax




if __name__=='__main__':
	with jax.disable_jit():
		import pick_and_run as p
		p.Run(**p.profile).run_as_main()
