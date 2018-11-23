

import yaml
import tensorflow as tf

from .args import get_args
from .estimator import get_estimator
from .input import gen_input_fn

if __name__ == "__main__":

	args = get_args()
	estimator = get_estimator(args)

	estimator.train(input_fn=gen_input_fn(args, "train"), max_steps=args["train_steps"])
	results = estimator.evaluate(input_fn=gen_input_fn(args, "eval"))

	try:
		with tf.gfile.GFile(args["results_path"], "r") as file:
			doc = yaml.load(file)
	except:
		doc = {}

	doc[args["name"]] = results

	with tf.gfile.GFile(args["results_path"], "w") as file:
		yaml.dump(doc, file)

