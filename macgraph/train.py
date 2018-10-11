
# from comet_ml import Experiment

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

	if args["use_comet"]:
		experiment = Experiment(api_key="bRptcjkrwOuba29GcyiNaGDbj", project_name="macgraph")
		experiment.log_multiple_params(args)

	estimator = get_estimator(args)

	if args["use_tf_debug"]:
		hooks = [tf_debug.LocalCLIDebugHook()]
	else:
		hooks = []

	train_spec = tf.estimator.TrainSpec(
		input_fn=gen_input_fn(args, "train"), 
		max_steps=args["max_steps"]*1000 if args["max_steps"] is not None else None,
		hooks=hooks)
	
	eval_spec  = tf.estimator.EvalSpec(
		input_fn=gen_input_fn(args, "eval"),
		throttle_secs=args["eval_every"])

	tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)



if __name__ == "__main__":
	args = get_args()

	# Logging setup
	logging.basicConfig()
	tf.logging.set_verbosity(args["log_level"])
	logger.setLevel(args["log_level"])
	logging.getLogger("mac-graph").setLevel(args["log_level"])

	# Info about the experiment, for the record
	train_size = sum(1 for _ in tf.python_io.tf_record_iterator(args["train_input_path"]))
	logger.info(args)
	logger.info(f"Training on {train_size} records")

	# DO IT!
	train(args)



