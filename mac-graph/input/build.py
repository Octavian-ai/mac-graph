
import tensorflow as tf

from .graph_util import *
from .text_util import *
from .util import *



# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def generate_record(args, vocab, doc):

	q = pretokenize_english(doc["question"]["english"])
	q = expand_unknown_vocab(q, vocab)
	q = string_to_tokens(q, vocab)

	a = doc["answer"]
	label = lookup_vocab(a, vocab)

	graph = graph_to_table(args, doc["graph"])

	feature = {
		"src": 				tf.train.Feature(int64_list=tf.train.Int64List(value=q)),
		"src_len": 			int64_feature(len(q)),
		"kb": 				tf.train.Feature(float_list=tf.train.FloatList(value=graph.flatten())),
		"kb_width": 		int64_feature(args["kb_width"]),
		"label": 			int64_feature(label),
	}

	example = tf.train.Example(features=tf.train.Features(feature=feature))
	return example.SerializeToString()



# --------------------------------------------------------------------------
# Run the script
# --------------------------------------------------------------------------

if __name__ == "__main__":

	def extras(parser):
		parser.add_argument('--skip-extract', action='store_true')
		parser.add_argument('--gqa_path', type=str, default="./input/gqa.yaml")

	args = get_args(extras)

	build_vocab(args)
	vocab = load_vocab(args)

	with Partitioner(args) as p:
		for i in read_gqa(args):
			p.write(generate_record(args, vocab, i))




