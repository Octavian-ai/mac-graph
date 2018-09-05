
import tensorflow as tf
import numpy as np

from collections import Counter
from tqdm import tqdm
import tableprint as tp
import re

from .args import *
from .util import *
from .text_util import *
from .input import *

def eager_to_str(v):
	return bytes_to_string(np.array(v))

def extend_args(parser):
	parser.add_argument('--print-records', action='store_true')

if __name__ == "__main__":

	args = get_args(extend_args)

	vocab = Vocab.load(args)
	count = 0
	tf.enable_eager_execution()

	dist = Counter()
	types = set()
	labels = set()


	for i in tqdm(tf.python_io.tf_record_iterator(args["train_input_path"]), total=args["limit"]):

		if args["limit"] is not None and count > args["limit"]:
			break

		# Parse
		r = parse_single_example(i)
		r,label = reshape_example(args, r)
		r["type_string"] = eager_to_str(r["type_string"])
		r["src"] = vocab.ids_to_english(np.array(r["src"]))
		r["label"] = vocab.inverse_lookup(int(r["label"]))
		r["kb_nodes"] = [vocab.ids_to_english(np.array(i)) for i in r["kb_nodes"] if np.array(i).size > 0]

		count += 1

		# Skip non matching prefixes
		if args["type_string_prefix"] is not None:
			if not r["type_string"].startswith(args["type_string_prefix"]):
				continue

		types.add(r["type_string"])
		labels.add(r["label"])

		dist[(r["label"], r["type_string"])] += 1
		
		if args["print_records"]:
			print(r["src"] + " = " + r["label"])
			for j in r["kb_nodes"]:
				print("NODE: " + j)
			print()
		
	

	print(f"\nTotal records processed: {count}")

	def shorten(i):
		return re.sub('[^A-Z]', '', i)
		# return i.replace("Station", "S").replace("Property", "P").replace("Adjacent", "A")

	headers = ["Label"] + [shorten(i) for i in list(types)] + ["Total"]
	data = [ [label] + [dist[(label, tpe)] for tpe in types] + [sum([dist[(label, tpe)] for tpe in types])] for label in labels]
	data.append(["Total"] + [sum([dist[(label, tpe)] for label in labels]) for tpe in types] + [sum(dist.values())])
	width = [20] + [7 for i in types] + [7]
	tp.table(data, headers, width=width)




