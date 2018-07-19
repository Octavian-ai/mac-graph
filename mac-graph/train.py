
import tensorflow as tf

from .estimator import get_estimator
from .input import gen_input_fn
from .args import get_args
from .predict import predict

# Make TF be quiet
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import logging
logger = logging.getLogger(__name__)



def train(args):

	estimator = get_estimator(args)

	train_spec = tf.estimator.TrainSpec(input_fn=gen_input_fn(args, "train"), max_steps=args["max_steps"])
	eval_spec  = tf.estimator.EvalSpec(input_fn=gen_input_fn(args, "eval"))

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
	predict(args)



