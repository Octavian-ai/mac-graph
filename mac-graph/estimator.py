
import tensorflow as tf

from .model import model_fn

def get_estimator(args):
	return tf.estimator.Estimator(model_fn, 
		model_dir=args["model_dir"],
		warm_start_from=args["warm_start_dir"],
		params=args)