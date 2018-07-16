
import tensorflow as tf
import numpy as np
from collections import Counter

from .args import get_args
from .estimator import get_estimator
from .input import *


def predict(args):
	estimator = get_estimator(args)
	predictions = estimator.predict(input_fn=gen_input_fn(args, "predict"))

	vocab   = Vocab.load(args)
	stats = Counter()
	answer_classes = Counter()
	predicted_classes = Counter()

	# print("Failed predictions:")

	for p in predictions:

		type_string = vocab.prediction_value_to_string(p["type_string"])
		answer_classes[vocab.prediction_value_to_string(p["actual_label"])] += 1
		predicted_classes[vocab.prediction_value_to_string(p["predicted_label"])] += 1


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

	print(f"\nStats:")
	for k, v in stats.items():
		print(f"{k}: {v}")

	print(f"\nPredicted classes:")
	for k, v in predicted_classes.items():
		print(f"{k}: {v}")

	print(f"\nAnswer classes:")
	for k, v in answer_classes.items():
		print(f"{k}: {v}")


if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.WARN)
	args = get_args()
	predict(args)



