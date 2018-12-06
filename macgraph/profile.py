
try:
	# import comet_ml in the top of your file
	from comet_ml import Experiment

except:
	# It's ok if we didn't install it
	pass

import tensorflow as tf

from .estimator import get_estimator
from .input import gen_input_fn
from .args import *



if __name__ == "__main__":

	steps = 100

	args = get_args()
	save_args(args)

	estimator = get_estimator(args)

	opts = (tf.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder.time_and_memory()) 
		.with_max_depth(15)
		.with_min_execution_time(min_micros=500)
		.build())

	# Collect traces of steps 10~20, dump the whole profile (with traces of
	# step 10~20) at step 20. The dumped profile can be used for further profiling
	# with command line interface or Web UI.

	with tf.contrib.tfprof.ProfileContext(args["model_dir"], dump_steps=[steps-10]) as pctx:

		# Run online profiling with 'op' view and 'opts' options at step 15, 18, 20.
		pctx.add_auto_profiling('op', opts, [steps-10])
		pctx.add_auto_profiling('graph', opts, [steps-10])
		pctx.add_auto_profiling('code', opts, [steps-10])
		
		estimator.train(input_fn=gen_input_fn(args, "train"), steps=steps)