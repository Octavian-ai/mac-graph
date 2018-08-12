
import argparse
import os.path
import yaml
import pathlib
import tensorflow as tf

from .minception import mi_activation

global_args = {}


# Expand activation args to callables
activation_options = {
	"tanh": tf.tanh,
	"relu": tf.nn.relu,
	"sigmoid": tf.nn.sigmoid,
	"mi": mi_activation,
	"abs": lambda x: tf.nn.relu(x) + tf.nn.relu(-x),
}

def generate_args_derivatives(args):

	r = {}
	r["modes"] = ["eval", "train", "predict"]

	# Expand input dirs
	for i in [*r["modes"], "all"]:
		r[i+"_input_path"] = os.path.join(args["input_dir"], i+"_input.tfrecords")

	r["vocab_path"] = os.path.join(args["input_dir"], "vocab.txt")
	r["config_path"] = os.path.join(args["model_dir"], "config.yaml")
	r["question_types_path"] = os.path.join(args["input_dir"], "types.yaml")
	r["answer_classes_path"] = os.path.join(args["input_dir"], "answer_classes.yaml")

	# Expand activation args to callables
	act_args = [key for key, value in args.items() if key.endswith("_activation") and isinstance(value, str)]
	for i in act_args:
		r[i] = activation_options[args[i].lower()]

	r["input_width"] = args["embed_width"] * 2

	return r

def get_args(extend=lambda parser:None, argv=None):

	parser = argparse.ArgumentParser()
	extend(parser)

	# --------------------------------------------------------------------------
	# General
	# --------------------------------------------------------------------------

	parser.add_argument('--log-level',  				type=str, default='INFO')
	parser.add_argument('--output-dir', 				type=str, default="./output")
	parser.add_argument('--input-dir',  				type=str, default="./input_data/processed/default")
	parser.add_argument('--model-dir',      			type=str, default="./output/model/default")

	# Used in train / predict / build
	parser.add_argument('--limit',						type=int, default=None, help="How many rows of input data to read")
	parser.add_argument('--type-string-prefix',			type=str, default=None, help="Filter input data rows to only have this type string prefix")

	# --------------------------------------------------------------------------
	# Data build
	# --------------------------------------------------------------------------

	parser.add_argument('--eval-holdback',    			type=float, default=0.1)
	parser.add_argument('--predict-holdback', 			type=float, default=0.005)


	# --------------------------------------------------------------------------
	# Training
	# --------------------------------------------------------------------------

	parser.add_argument('--warm-start-dir',				type=str, default=None, help="Load model initial weights from previous checkpoints")
	
	parser.add_argument('--batch-size',            		type=int, default=32,   help="Number of items in a full batch")
	parser.add_argument('--max-steps',             		type=int, default=None)
		
	parser.add_argument('--max-gradient-norm',     		type=float, default=30.0)
	parser.add_argument('--learning-rate',         		type=float, default=2.4E-3)

	# --------------------------------------------------------------------------
	# Network topology
	# --------------------------------------------------------------------------

	parser.add_argument('--vocab-size',	           		type=int, default=128,   help="How many different words are in vocab")
	parser.add_argument('--answer-classes',	       		type=int, default=128,    help="The number of different possible answers (e.g. answer classes). Currently tied to vocab size since we attempt to tokenise the output.")
	
	parser.add_argument('--max-seq-len',	  	 		type=int, default=20,   help="Maximum length of question token list")
	
	parser.add_argument('--input-layers',	   			type=int, default=3,    help="How many input layers are in the english encoding LSTM stack")
	parser.add_argument('--input-dropout',              type=float, default=0.2)
	parser.add_argument('--embed-width',	       		type=int, default=64,   help="The width of token embeddings")
	
	parser.add_argument('--kb-node-width',         		type=int, default=7,    help="Width of node entry into graph table aka the knowledge base")
	parser.add_argument('--kb-node-max-len',         	type=int, default=40,   help="Width of node entry into graph table aka the knowledge base")
	parser.add_argument('--kb-edge-width',         		type=int, default=3,    help="Width of edge entry into graph table aka the knowledge base")
	parser.add_argument('--kb-edge-max-len',         	type=int, default=40,    help="Width of edge entry into graph table aka the knowledge base")
	
	parser.add_argument('--read-heads',         		type=int, default=1,    help="Number of read heads for each knowledge base table")
	parser.add_argument('--read-indicator-rows',        type=int, default=0,    help="Number of extra trainable rows")
	parser.add_argument('--read-indicator-cols',        type=int, default=0,    help="Number of extra trainable rows")
	parser.add_argument('--read-dropout',         		type=float, default=0.2,    help="Dropout on read heads")
	parser.add_argument('--read-activation',			type=str, default="mi")
	parser.add_argument('--read-from-question',			action='store_true')

	parser.add_argument('--data-stack-width',         	type=int, default=64,   help="Width of stack entry")
	parser.add_argument('--data-stack-len',         	type=int, default=20,   help="Length of stack")
	
	parser.add_argument('--control-width',	           	type=int, default=64,	help="The width of control state")
	parser.add_argument('--control-heads',	           	type=int, default=1,	help="The number of control question-word attention heads")
	parser.add_argument('--control-dropout',	        type=float, default=0.2, help="Dropout on the control unit")

	parser.add_argument('--memory-width',	           	type=int, default=64,	help="The width of memory state")
	parser.add_argument('--memory-transform-layers',	type=int, default=2, 	help="How many deep layers in memory transforms")

	parser.add_argument('--output-activation',			type=str, default="mi")
	parser.add_argument('--output-layers',				type=int, default=2)

	
	parser.add_argument('--disable-kb-node', 			action='store_false', dest='use_kb_node')
	parser.add_argument('--disable-kb-edge', 			action='store_false', dest='use_kb_edge')
	parser.add_argument('--enable-data-stack', 			action='store_true',  dest='use_data_stack')
	parser.add_argument('--enable-attn-score-dense', 	action='store_true',  dest='use_attn_score_dense')
	parser.add_argument('--enable-position-encoding', 	action='store_true',  dest='use_position_encoding')
	parser.add_argument('--disable-control-cell', 		action='store_false', dest="use_control_cell")
	parser.add_argument('--disable-memory-cell', 		action='store_false', dest="use_memory_cell")
	parser.add_argument('--disable-dynamic-decode', 	action='store_false', dest="use_dynamic_decode")
	parser.add_argument('--disable-question-state', 	action='store_false', dest="use_question_state")
	parser.add_argument('--disable-read-abs', 			action='store_false', dest="use_read_abs")
	parser.add_argument('--enable-lr-finder', 			action='store_true',  dest="use_lr_finder")
	parser.add_argument('--enable-lr-decay', 			action='store_true',  dest="use_lr_decay")

	parser.add_argument('--enable-tf-debug', 			action='store_true',  dest="use_tf_debug")
	parser.add_argument('--enable-comet', 				action='store_true',  dest="use_comet")

	parser.add_argument('--max-decode-iterations', 		type=int, default=8)
	
	args = vars(parser.parse_args(argv))


	args.update(generate_args_derivatives(args))
	
	# Global singleton var for easy access deep in the codebase (e.g. utility functions)
	# Note that this wont play well with PBT!! 
	# TODO: Remove
	global_args.clear()
	global_args.update(args)

	return args


def save_args(args):
	pathlib.Path(args["model_dir"]).mkdir(parents=True, exist_ok=True)
	with tf.gfile.GFile(os.path.join(args["config_path"]), "w") as file:
		yaml.dump(args, file)
