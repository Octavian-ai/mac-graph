
try:
	# import comet_ml in the top of your file
	from comet_ml import Experiment

except:
	# It's ok if we didn't install it
	pass

import tensorflow as tf

from .args import get_args
from .train import train



if __name__ == "__main__":
	args = get_args()
	with tf.contrib.tfprof.ProfileContext(args["profile_path"]) as pctx:
		train(args)