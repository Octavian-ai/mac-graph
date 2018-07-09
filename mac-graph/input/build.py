
import tensorflow as tf

from .graph_util import *
from .text_util import *
from .util import *
from ..args import *

import logging
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def generate_record(args, vocab, doc):

	q = vocab.english_to_ids(doc["question"]["english"])

	# May raise exception if unsupported type
	label = vocab.lookup(pretokenize_json(doc["answer"]))

	if label == UNK_ID:
		raise ValueError("We're only including questions that have in-vocab answers")

	graph = graph_to_table(args, vocab, doc["graph"])

	logger.debug(f"Record: {vocab.ids_to_string([label])}, {vocab.ids_to_string(q)}, {[vocab.ids_to_string(g) for g in graph]}")

	feature = {
		"src": 				tf.train.Feature(int64_list=tf.train.Int64List(value=q)),
		"src_len": 			int64_feature(len(q)),
		"kb": 				tf.train.Feature(int64_list=tf.train.Int64List(value=graph.flatten())),
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
		parser.add_argument('--gqa_path', type=str, default="./input_raw/gqa.yaml")

	args = get_args(extras)

	logging.basicConfig()
	logger.setLevel(args["log_level"])

	logger.info("Build vocab")
	vocab = Vocab.build(args, lambda i:gqa_to_tokens(args, i))
	logger.debug(f"vocab: {vocab.table}")

	written = 0

	logger.info("Generate TFRecords")
	with Partitioner(args) as p:
		for i in read_gqa(args):
			try:
				p.write(generate_record(args, vocab, i))
				written += 1
			except ValueError:
				pass

	logger.info(f"Wrote {written} TFRecords")





