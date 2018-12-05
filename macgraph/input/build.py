
import tensorflow as tf
import pathlib
from collections import Counter
import yaml
from tqdm import tqdm
import contextlib

from .graph_util import *
from .text_util import *
from .util import *
from .args import *
from .balancer import TwoLevelBalancer
from .partitioner import *

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
		raise ValueError(f"We're only including questions that have in-vocab answers ({doc['answer']})")

	if label >= args["output_width"]:
		raise ValueError(f"Label {label} greater than answer classes {args['output_width']}")

	nodes, edges, adjacency = graph_to_table(args, vocab, doc["graph"])

	logger.debug(f"""
Answer={vocab.ids_to_string([label])} 
{vocab.ids_to_string(q)}
{[vocab.ids_to_string(g) for g in nodes]}
{[vocab.ids_to_string(g) for g in edges]}""")

	feature = {
		"src": 					write_int64_array_feature(q),
		"src_len": 				write_int64_feature(len(q)),
		"kb_edges": 			write_int64_array_feature(edges.flatten()),
		"kb_edges_len": 		write_int64_feature(edges.shape[0]),
		"kb_nodes": 			write_int64_array_feature(nodes.flatten()),
		"kb_nodes_len": 		write_int64_feature(nodes.shape[0]),		
		"kb_adjacency":			write_boolean_array_feature(adjacency.flatten()),
		"label": 				write_int64_feature(label),
		"type_string":			write_string_feature(doc["question"]["type_string"]),
	}

	example = tf.train.Example(features=tf.train.Features(feature=feature))
	return example.SerializeToString()


def build(args):
	try:
		pathlib.Path(args["input_dir"]).mkdir(parents=True, exist_ok=True)
	except FileExistsError:
		pass

	logger.info(f"Building {args['dataset']} data from {args['gqa_paths']}")

	if not args["skip_vocab"]:
		logger.info(f"Build vocab {args['vocab_path']} ")
		vocab = Vocab.build(args, lambda i:gqa_to_tokens(args, i), limit=min(args["limit"], args["vocab_build_limit"]))
		logger.info(f"Wrote {len(vocab)} vocab entries")
		logger.debug(f"vocab: {vocab.table}")
		print()

		if args["only_build_vocab"]:
			return
	else:
		vocab = Vocab.load_from_args(args)


	question_types = Counter()
	output_classes = Counter()

	logger.info(f"Generate TFRecords {args['input_dir']}")

	# To close everything nicely later
	with ExitStack() as stack:

		balancers = {}

		k_answer = lambda d: d["answer"]
		k_type_string = lambda d: d["question"]["type_string"]

		for mode in args["modes"]:
			writer = stack.enter_context(RecordWriter(args, mode))
			balancers[mode] = stack.enter_context(TwoLevelBalancer(k_answer, k_type_string, writer, min_none(args["balance_batch"], args["limit"])))

		with Partitioner(args, balancers) as p:
			for doc in tqdm(read_gqa(args), total=args["limit"]):
				try:
					record = generate_record(args, vocab, doc)
					question_types[doc["question"]["type_string"]] += 1
					output_classes[doc["answer"]] += 1
					p.write(doc, record)

				except ValueError as ex:
					logger.debug(ex)
					pass


			with tf.gfile.GFile(args["answer_classes_path"], "w") as file:
				yaml.dump(dict(p.answer_classes), file)

			with tf.gfile.GFile(args["answer_classes_types_path"], "w") as file:
				yaml.dump(dict(p.answer_classes_types), file)

			logger.info(f"Class distribution: {p.answer_classes}")

			logger.info(f"Wrote {p.written} TFRecords")

		
	with tf.gfile.GFile(args["question_types_path"], "w") as file:
		yaml.dump(dict(question_types), file)


# --------------------------------------------------------------------------
# Run the script
# --------------------------------------------------------------------------

if __name__ == "__main__":

	args = get_args()

	logging.basicConfig()
	logger.setLevel(args["log_level"])
	logging.getLogger("mac-graph.input.util").setLevel(args["log_level"])

	build(args)

	





