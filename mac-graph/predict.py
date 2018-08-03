
import tensorflow as tf
import numpy as np
from collections import Counter

from .model import model_fn
from .input import *
from .args import get_args



def predict(args):
	estimator = tf.estimator.Estimator(model_fn, model_dir=args["model_dir"], params=args)
	predictions = estimator.predict(input_fn=gen_input_fn(args, "predict"))

	vocab   = Vocab.load(args)
	stats = Counter()

	print("Failed predictions:")

	for p in predictions:

		type_string = vocab.prediction_value_to_string(p["type_string"])

		if p["predicted_label"] == p["actual_label"]:
			stats["correct"] += 1
			stats["correct_"+type_string] += 1
		
		else:
			stats["incorrect"] += 1
			stats["incorrect_"+type_string] += 1


			for k, v in p.items():
				s = vocab.prediction_value_to_string(v)
				print(f"{k}: {s}")
			print("-------")

	print(f"Stats:")
	for k, v in stats.items():
		print(f"{k}: {v}")


if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.WARN)
	args = get_args()
	predict(args)



