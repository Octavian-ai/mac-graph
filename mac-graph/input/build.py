
import tensorflow as tf

from .build_text_data  import *
from .build_graph_data import *
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

	graph = graph_to_table(doc["graph"])

	feature = {
		"src": _conv_bytes_feature(q),
		"src_len": _int64_feature(len(q)),
		"knowledge_base": _conv_bytes_feature(graph),
		"label": _int64_feature(a),
	}

	example = tf.train.Example(features=tf.train.Features(feature=feature))
	return example.SerializeToString()



# --------------------------------------------------------------------------
# Run the script
# --------------------------------------------------------------------------

if __name__ == "__main__":

	def extras(parser):
		parser.add_argument('--skip-extract', action='store_true')
		parser.add_argument('--gqa_path', type=str, default="./data/gqa.yaml")

	args = get_args(extras)

	build_vocab(args)
	vocab = load_vocab(args)

	with Partitioner(args) as p:
		for i in read_gqa(args):
			p.write(generate_record(args, i, vocab))




