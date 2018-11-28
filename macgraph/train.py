
try:
	# import comet_ml in the top of your file
	from comet_ml import Experiment

except:
	# It's ok if we didn't install it
	pass

import tensorflow as tf
from tensorflow.python import debug as tf_debug

from .estimator import get_estimator
from .input import gen_input_fn
from .args import *

# Make TF be quiet
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import logging
logger = logging.getLogger(__name__)



def train(args):

	# So I don't frigging forget what caused working models
	save_args(args)
	

	if args["use_tf_debug"]:
		hooks = [tf_debug.LocalCLIDebugHook()]
	else:
		hooks = []

	if args["use_comet"]:
		# Add the following code anywhere in your machine learning file
		experiment = Experiment(api_key="bRptcjkrwOuba29GcyiNaGDbj", project_name="macgraph", workspace="davidhughhenrymack")
		experiment.log_multiple_params(args)

		if len(args["tag"]) > 0:
			experiment.add_tags(args["tag"])


	train_size = sum(1 for _ in tf.python_io.tf_record_iterator(args["train_input_path"]))
	logger.info(f"Training on {train_size} records")

	# ----------------------------------------------------------------------------------

	estimator = get_estimator(args)

	train_spec = tf.estimator.TrainSpec(
		input_fn=gen_input_fn(args, "train"), 
		max_steps=args["train_max_steps"]*1000 if args["train_max_steps"] is not None else None,
		hooks=hooks)
	
	eval_spec  = tf.estimator.EvalSpec(
		input_fn=gen_input_fn(args, "eval"),
		throttle_secs=args["eval_every"])

	tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)



if __name__ == "__main__":
	args = get_args()

	# DO IT!
	train(args)



