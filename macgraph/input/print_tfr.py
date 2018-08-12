
import tensorflow as tf
import numpy as np

from .args import *
from .util import *
from .text_util import Vocab
from .input import *

if __name__ == "__main__":

	args = get_args()

	vocab = Vocab.load(args)
	count = 0
	tf.enable_eager_execution()

	# Info about the experiment, for the record
	for i in tf.python_io.tf_record_iterator(args["train_input_path"]):
		r = parse_single_example(i)
		r,label = reshape_example(args, r)

		r["src"] = vocab.ids_to_english(np.array(r["src"]))
		r["label"] = vocab.inverse_lookup(int(r["label"]))
		r["kb_nodes"] = [vocab.ids_to_english(np.array(i)) for i in r["kb_nodes"] if np.array(i).size > 0]

		print(r["src"] + " = " + r["label"])
		for j in r["kb_nodes"]:
			print("NODE: " + j)
		print()

		count += 1

		if args["limit"] is not None and count > args["limit"]:
			break
