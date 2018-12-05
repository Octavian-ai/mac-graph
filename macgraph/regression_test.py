
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
from .train import train

# Make TF be quiet
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

if __name__ == "__main__":

	args = get_args()
	train(args)

	estimator = get_estimator(args)
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

	print(args["dataset"], simplified_results)

	if not args["dataset"] in doc or simplified_results["accuracy"] > doc[args["dataset"]]["accuracy"]:
		doc[args["dataset"]] = simplified_results

	with tf.gfile.GFile(args["results_path"], "w") as file:
		yaml.dump(doc, file)

