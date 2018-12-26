
try:
	# import comet_ml in the top of your file
	from comet_ml import Experiment

except:
	# It's ok if we didn't install it
	pass

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from collections import namedtuple

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
		experiment.log_parameters(args)

		if len(args["tag"]) > 0:
			experiment.add_tags(args["tag"])


	train_size = sum(1 for _ in tf.python_io.tf_record_iterator(args["train_input_path"]))
	tf.logging.info(f"Training on {train_size} records")

	# ----------------------------------------------------------------------------------

	

	training_segments = []
	TrainingSegment = namedtuple('TrainingSegment', ['args', 'max_steps'])

	if args["use_curriculum"]:
		assert args["train_max_steps"] is not None, "Curriculum training requires --train-max-steps"

		seg_steps = args["train_max_steps"] / float(args["max_decode_iterations"])

		for i in range(1, args["max_decode_iterations"]+1):

			seg_args = {**args}
			seg_args["filter_output_class"] = [str(j) for j in list(range(i+1))]
			total_seg_steps = i*seg_steps*1000

			
			training_segments.append(TrainingSegment(seg_args, total_seg_steps))

	else:
		training_segments.append(TrainingSegment(args, args["train_max_steps"]*1000 if args["train_max_steps"] is not None else None))


	for i in training_segments:

		tf.logging.info(f"Begin training segment {i.max_steps} {i.args['filter_output_class']}")

		estimator = get_estimator(i.args)

		train_spec = tf.estimator.TrainSpec(
			input_fn=gen_input_fn(i.args, "train"), 
			max_steps=int(i.max_steps),
			hooks=hooks)
		
		eval_spec  = tf.estimator.EvalSpec(
			input_fn=gen_input_fn(i.args, "eval"),
			throttle_secs=i.args["eval_every"])

		tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)



if __name__ == "__main__":
	args = get_args()

	# DO IT!
	train(args)



