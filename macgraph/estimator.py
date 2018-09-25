
import tensorflow as tf

from .model import model_fn

def get_estimator(args):

	run_config = tf.estimator.RunConfig(
		model_dir=args["model_dir"],
		tf_random_seed=3,
		save_checkpoints_steps=args["save_checkpoints_steps"],
		save_checkpoints_secs=args["save_checkpoints_secs"],
	)

	return tf.estimator.Estimator(
		model_fn=model_fn, 
		config=run_config,
		warm_start_from=args["warm_start_dir"],
		params=args)