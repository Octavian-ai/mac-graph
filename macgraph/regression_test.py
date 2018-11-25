
try:
	# import comet_ml in the top of your file
	from comet_ml import Experiment

except:
	# It's ok if we didn't install it
	pass

import yaml
import tensorflow as tf

from .args import get_args
from .estimator import get_estimator
from .input import gen_input_fn

# Make TF be quiet
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

if __name__ == "__main__":

	args = get_args()
	estimator = get_estimator(args)

	if args["use_comet"]:
		# Add the following code anywhere in your machine learning file
		experiment = Experiment(api_key="bRptcjkrwOuba29GcyiNaGDbj", project_name="macgraph", workspace="davidhughhenrymack")
		experiment.log_multiple_params(args)


	train_spec = tf.estimator.TrainSpec(
		input_fn=gen_input_fn(args, "train"), 
		max_steps=args["train_steps"]*1000)
	
	eval_spec  = tf.estimator.EvalSpec(
		input_fn=gen_input_fn(args, "eval"),
		throttle_secs=args["eval_every"])

	tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

	results = estimator.evaluate(input_fn=gen_input_fn(args, "eval"))

	try:
		with tf.gfile.GFile(args["results_path"], "r") as file:
			doc = yaml.load(file)
	except:
		doc = {}

	if doc is None:
		doc = {}


	simplified_results = {
		"accuracy": float(results["accuracy"]),
		"accuracy_pct": str(round(results["accuracy"]*1000.0)/10.0) + "%",
		"loss": float(results["loss"]),
		"steps": int(results["current_step"]),
	}

	print(args["name"], simplified_results)
	doc[args["name"]] = simplified_results

	if args["use_comet"]:
		experiment.log_multiple_metrics(simplified_results)

	with tf.gfile.GFile(args["results_path"], "w") as file:
		yaml.dump(doc, file)

