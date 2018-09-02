
import tensorflow as tf
import pathlib
from collections import Counter
import yaml
from tqdm import tqdm

from .graph_util import *
from .text_util import *
from .util import *
from .args import *
from .balancer import TwoLevelBalancer

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

	if label >= args["output_classes"]:
		raise ValueError(f"Label {label} greater than answer classes {args['output_classes']}")

	nodes, edges = graph_to_table(args, vocab, doc["graph"])

	logger.debug(f"""
Answer={vocab.ids_to_string([label])} 
{vocab.ids_to_string(q)}
{[vocab.ids_to_string(g) for g in nodes]}
{[vocab.ids_to_string(g) for g in edges]}""")

	feature = {
		"src": 				tf.train.Feature(int64_list=tf.train.Int64List(value=q)),
		"src_len": 			int64_feature(len(q)),
		"kb_edges": 		tf.train.Feature(int64_list=tf.train.Int64List(value=edges.flatten())),
		"kb_edges_len": 	int64_feature(edges.shape[0]),
		"kb_nodes": 		tf.train.Feature(int64_list=tf.train.Int64List(value=nodes.flatten())),
		"kb_nodes_len": 	int64_feature(nodes.shape[0]),		
		"label": 			int64_feature(label),
		"type_string":		string_feature(doc["question"]["type_string"]),
	}

	example = tf.train.Example(features=tf.train.Features(feature=feature))
	return example.SerializeToString()



# --------------------------------------------------------------------------
# Run the script
# --------------------------------------------------------------------------

if __name__ == "__main__":

	args = get_args()

	logging.basicConfig()
	logger.setLevel(args["log_level"])
	logging.getLogger("mac-graph.input.util").setLevel(args["log_level"])

	pathlib.Path(args["input_dir"]).mkdir(parents=True, exist_ok=True)

	if not args["skip_vocab"]:
		logger.info("Build vocab")
		vocab = Vocab.build(args, lambda i:gqa_to_tokens(args, i))
		logger.info(f"Wrote {len(vocab)} vocab entries")
		logger.debug(f"vocab: {vocab.table}")
		print()
	else:
		vocab = Vocab.load(args)


	question_types = Counter()
	output_classes = Counter()

	logger.info("Generate TFRecords")
	with Partitioner(args) as p:
		with TwoLevelBalancer(lambda d: d["answer"], lambda d: d["question"]["type_string"], p, args["balance_batch"]) as balancer:
			for doc in tqdm(read_gqa(args)):
				try:
					record = generate_record(args, vocab, doc)
					question_types[doc["question"]["type_string"]] += 1
					output_classes[doc["answer"]] += 1
					balancer.add(doc, record)

				except ValueError as ex:
					logger.debug(ex)
					pass


		with tf.gfile.GFile(args["answer_classes_path"], "w") as file:
			yaml.dump(dict(p.answer_classes), file)

		with tf.gfile.GFile(args["answer_classes_types_path"], "w") as file:
			yaml.dump(dict(p.answer_classes_types), file)

		logger.info(f"Wrote {p.written} TFRecords")

		
	with tf.gfile.GFile(args["question_types_path"], "w") as file:
		yaml.dump(dict(question_types), file)

	





