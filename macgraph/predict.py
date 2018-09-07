
import tensorflow as tf
import numpy as np
from collections import Counter

from .args import get_args
from .estimator import get_estimator
from .input import *

def extend_args(parser):
	parser.add_argument("--n-detail-rows",type=int,default=15)


def predict(args):
	estimator = get_estimator(args)
	predictions = estimator.predict(input_fn=gen_input_fn(args, "predict"))
	vocab = Vocab.load(args)

	def print_row(row):
		for k, v in row.items():
			print(f"{k}: {v}")
		print("-------")


	def decode_row(row):
		for i in ["type_string", "actual_label", "predicted_label", "src"]:
			row[i] = vocab.prediction_value_to_string(row[i])

	stats = Counter()
	output_classes = Counter()
	predicted_classes = Counter()
	confusion = Counter()

	for count, p in enumerate(predictions):
		decode_row(p)
		if args["type_string_prefix"] is None or p["type_string"].startswith(args["type_string_prefix"]):

			output_classes[p["actual_label"]] += 1
			predicted_classes[p["predicted_label"]] += 1

			if p["actual_label"] == p["predicted_label"]:
				emoji = "✅"
			else:
				emoji = "❌"

			confusion[emoji + " \texp:" + p["actual_label"] +" \tact:" + p["predicted_label"] + " \t" + p["type_string"]] += 1

			if count <= args["n_detail_rows"]:
				print_row(p)

	print(f"\nConfusion matrix:")
	for k, v in confusion.most_common():
		print(f"{k}: {v}")



if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.WARN)
	args = get_args(extend_args)
	predict(args)



