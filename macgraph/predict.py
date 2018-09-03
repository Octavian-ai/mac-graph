
import tensorflow as tf
import numpy as np
from collections import Counter

from .args import get_args
from .estimator import get_estimator
from .input import *


def predict(args):
	estimator = get_estimator(args)
	predictions = estimator.predict(input_fn=gen_input_fn(args, "predict"))
	vocab = Vocab.load(args)

	stats = Counter()
	output_classes = Counter()
	predicted_classes = Counter()
	confusion = Counter()

	for p in predictions:

		type_string = vocab.prediction_value_to_string(p["type_string"])
		answer_string = vocab.prediction_value_to_string(p["actual_label"])
		predicted_label = vocab.prediction_value_to_string(p["predicted_label"])
		
		output_classes[answer_string] += 1
		predicted_classes[predicted_label] += 1

		if answer_string == predicted_label:
			emoji = "✅"
		else:
			emoji = "❌"

		confusion[emoji + " \texp:"+answer_string+" \tact:" + predicted_label + " \t"+type_string] += 1

		# for k, v in p.items():
		# 	s = vocab.prediction_value_to_string(v)
		# 	print(f"{k}: {s}")
		# print("-------")

	print(f"\nConfusion matrix:")
	for k, v in confusion.most_common():
		print(f"{k}: {v}")

	# print(f"\nPredicted classes:")
	# for k, v in predicted_classes.most_common():
	# 	print(f"{k}: {v}")

	# print(f"\nAnswer classes:")
	# for k, v in output_classes.most_common():
	# 	print(f"{k}: {v}")




if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.WARN)
	args = get_args()
	predict(args)



