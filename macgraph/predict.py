
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
	first_station_incorrect = Counter()
	second_station_incorrect = Counter()


	# print("Failed predictions:")

	for p in predictions:

		type_string = vocab.prediction_value_to_string(p["type_string"])
		answer_string = vocab.prediction_value_to_string(p["actual_label"])
		answer_classes[answer_string] += 1
		predicted_classes[vocab.prediction_value_to_string(p["predicted_label"])] += 1

		# Are 16 and 22 adjacent?
		# first_station = vocab.inverse_lookup(p["src"][2])
		# second_station = vocab.inverse_lookup(p["src"][6])

		if p["predicted_label"] == p["actual_label"]:
			stats["correct"] += 1
			stats["correct_"+type_string] += 1
			stats["correct_"+answer_string] += 1
			# stats[f"correct_pair_{first_station}_{second_station}"] += 1
		
		else:
			stats["incorrect"] += 1
			stats["incorrect_"+type_string] += 1
			stats["incorrect_"+answer_string] += 1

			# first_station_incorrect[first_station] += 1
			# first_station_incorrect[second_station] += 1
			# stats[f"incorrect_pair_{first_station}_{second_station}"] += 1


		for k, v in p.items():
			s = vocab.prediction_value_to_string(v)
			print(f"{k}: {s}")
		print("-------")

	print(f"\nStats:")
	for k, v in stats.most_common():
		print(f"{k}: {v}")

	print(f"\nPredicted classes:")
	for k, v in predicted_classes.most_common():
		print(f"{k}: {v}")

	print(f"\nAnswer classes:")
	for k, v in answer_classes.most_common():
		print(f"{k}: {v}")

	# print(f"\nstation_incorrect:")
	# for k, v in first_station_incorrect.most_common():
	# 	print(f"{k}: {v}")

	# print(f"\nsecond_station_incorrect:")
	# for k, v in second_station_incorrect.most_common():
	# 	print(f"{k}: {v}")




if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.WARN)
	args = get_args()
	predict(args)



