
import argparse
import os.path
import yaml
import subprocess
import pathlib
import tensorflow as tf
import glob
import logging
import coloredlogs

from .global_args import global_args

from .activations import ACTIVATION_FNS
from .input import Vocab



def generate_args_derivatives(args):

	r = {}
	r["modes"] = ["eval", "train", "predict"]

	if "gqa_paths" in args:
		if args["gqa_paths"] == [] or args["gqa_paths"] == None:
			r["gqa_paths"] = [os.path.join(args["gqa_dir"], args["dataset"]) + ".yaml"]
		else:
			gp = []
			for i in args["gqa_paths"]:
				if "*" in i:
					gp += glob.glob(i)
				else:
					gp.append(i)

			r["gqa_paths"] = gp
		
	if args["input_dir"] is None:
		r["input_dir"] = os.path.join(args["input_dir_prefix"], args["dataset"])
	else:
		r["input_dir"] = args["input_dir"]

	if args["model_dir"] is None:
		r["model_dir"] = os.path.join(args["model_dir_prefix"], args["dataset"], *args["tag"], args["model_version"], f"r_{args["random_seed"]}")
	else:
		r["model_dir"] = args["model_dir"]


	r["profile_path"] = os.path.join(r["model_dir"], "profile")

	# Expand input dirs
	for i in [*r["modes"], "all"]:
		r[i+"_input_path"] = os.path.join(r["input_dir"], i+"_input.tfrecords")

	if args["vocab_path"] is None:
		r["vocab_path"] = os.path.join(r["input_dir"], "vocab.txt")
	else:
		r["vocab_path"] = args["vocab_path"]

	r["config_path"] = os.path.join(r["model_dir"], "config.yaml")
	r["question_types_path"] = os.path.join(r["input_dir"], "types.yaml")
	r["answer_classes_path"] = os.path.join(r["input_dir"], "answer_classes.yaml")
	r["answer_classes_types_path"] = os.path.join(r["input_dir"], "answer_classes_types.yaml")

	if args["control_width"] is None:
		r["control_width"] = args["input_width"] * args["control_heads"]

	if not args["use_input_bilstm"]:
		r["input_width"] = args["embed_width"]

	r["kb_list"] = []
	if args["use_kb_node"]:
		r["kb_list"].append("kb_node")
	if args["use_kb_edge"]:
		r["kb_list"].append("kb_edge")


	r["mp_head_list"] = ["mp_write", "mp_read0"]

	r["query_sources"] = [ "token_index"] # "token_content",
	r["query_taps"] = ["switch_attn", "token_index_attn"] # "token_content_attn",
	# r["query_sources"].append("step_const")
	# r["query_taps"].append("step_const_signal")

	if args["use_read_previous_outputs"]:
		r["query_sources"].append("prev_output")
		r["query_taps"].append("prev_output_attn")


	if args["use_fast"]:
		r["use_summary_scalar"] = False
		r["use_summary_image"] = False
		r["use_assert"] = False



	try:
		r["vocab"] = Vocab.load(r["vocab_path"], args["vocab_size"])
	except tf.errors.NotFoundError:
		pass

	return r

def get_git_hash():
	try:
		result = subprocess.run(
			['git', '--no-pager', 'log', "--pretty=format:%h", '-n', '1'],
			stdout=subprocess.PIPE,
			check=True,
			universal_newlines=True
		)
		return result.stdout
	except subprocess.CalledProcessError:
		# Git was angry, oh well
		return "unknown"

def get_args(extend=lambda parser:None, argv=None):

	parser = argparse.ArgumentParser()
	extend(parser)

	# --------------------------------------------------------------------------
	# General
	# --------------------------------------------------------------------------

	parser.add_argument('--log-level',  				type=str, default='INFO')
	parser.add_argument('--output-dir', 				type=str, default="./output")
	parser.add_argument('--dataset',					type=str, default="default", help="Name of dataset")
	parser.add_argument('--input-dir',					type=str, default=None)
	parser.add_argument('--input-dir-prefix',  			type=str, default="./input_data/processed")
	parser.add_argument('--tag',						action="append")
	
	parser.add_argument('--model-dir',					type=str, default=None)
	parser.add_argument('--model-version',      		type=str, default=get_git_hash(), help="Model will be saved to a directory with this name, to assist with repeatable experiments")	
	parser.add_argument('--model-dir-prefix',      		type=str, default="./output")
	

	# Used in train / predict / build
	parser.add_argument('--limit',						type=int, default=None, help="How many rows of input data to read")
	parser.add_argument('--filter-type-prefix',			type=str, default=None, help="Filter input data rows to only have this type string prefix")
	parser.add_argument('--filter-output-class',		action="append", help="Filter input data rows to only have this output class")

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
	parser.add_argument('--train-max-steps',            type=float, default=None, help="In thousands")
	parser.add_argument('--results-path', 				type=str, default="./results.yaml")

	parser.add_argument('--learning-rate',         		type=float, default=1e-3)
	parser.add_argument('--enable-regularization',		action='store_true', dest='use_regularization')
	parser.add_argument('--regularization-factor',		type=float, default=0.0001)
	parser.add_argument('--random-seed',				type=int, default=3)
	parser.add_argument('--enable-gradient-clipping',	action='store_true', dest='use_gradient_clipping')
	parser.add_argument('--max-gradient-norm',     		type=float, default=0.4)

	parser.add_argument('--enable-gradient-norm-loss',	action='store_true', dest='use_gradient_norm_loss')
	parser.add_argument('--gradient-norm-loss-factor',  type=float, default=1e-3)

	parser.add_argument('--eval-every',					type=int,	default=7*60, help="Evaluate every X seconds")

	parser.add_argument('--fast',						action='store_true', dest='use_fast')

	# --------------------------------------------------------------------------
	# Decode
	# --------------------------------------------------------------------------
	
	parser.add_argument('--max-decode-iterations', 		type=int, default=1)
	parser.add_argument('--finished-steps-loss-factor',	type=float, default= 0.001)
	parser.add_argument('--enable-dynamic-decode', 		action='store_true', dest="use_dynamic_decode")
	parser.add_argument('--enable-independent-iterations', action='store_true', dest="use_independent_iterations")

	# --------------------------------------------------------------------------
	# Network topology
	# --------------------------------------------------------------------------

	parser.add_argument('--vocab-size',	           		type=int, default=128,   help="How many different words are in vocab")
	parser.add_argument('--vocab-path',					type=str, default=None,	 help="Custom vocab path")

	parser.add_argument('--max-seq-len',	  	 		type=int, default=40,   help="Maximum length of question token list")
	
	parser.add_argument('--input-layers',	   			type=int, 	default=3,    help="How many input layers are in the english encoding LSTM stack")
	parser.add_argument('--input-dropout',              type=float, default=0.2)
	parser.add_argument('--input-width',             	type=int, 	default=128)
	parser.add_argument('--disable-input-residual', 	action='store_false', dest='use_input_residual')
	parser.add_argument('--disable-input-bilstm', 		action='store_false', dest='use_input_bilstm')
	parser.add_argument('--disable-input-adjacency', 	action='store_false', dest='use_input_adjacency')

	parser.add_argument('--embed-width',	       		type=int, default=64,   help="The width of token embeddings")
	parser.add_argument('--enable-embed-const-eye',		action='store_true', dest='use_embed_const_eye')

	
	parser.add_argument('--kb-node-width',         		type=int, default=7,    help="Width of node entry into graph table aka the knowledge base")
	parser.add_argument('--kb-node-max-len',         	type=int, default=40,   help="Maximum number of nodes in kb")
	parser.add_argument('--kb-edge-width',         		type=int, default=3,    help="Width of edge entry into graph table aka the knowledge base")
	parser.add_argument('--kb-edge-max-len',         	type=int, default=40,   help="Maximum number of edges in kb")
	parser.add_argument('--disable-kb-node', 			action='store_false', dest='use_kb_node')
	parser.add_argument('--disable-kb-edge', 			action='store_false', dest='use_kb_edge')

	parser.add_argument('--disable-read-cell',			action='store_false', dest='use_read_cell')
	parser.add_argument('--enable-read-previous-outputs',action='store_true', dest='use_read_previous_outputs')
	parser.add_argument('--read-width',         		type=int, default=128,  help="Width of the read state output")
	parser.add_argument('--read-heads',         		type=int, default=1,    help="Number of read heads for each knowledge base table")
	parser.add_argument('--read-layers',         		type=int, default=1,    help="Number of read transformation layers")
	parser.add_argument('--read-indicator-rows',        type=int, default=0,    help="Number of extra trainable rows")
	parser.add_argument('--read-indicator-cols',        type=int, default=0,    help="Number of extra trainable rows")
	parser.add_argument('--read-dropout',         		type=float, default=0.0,    help="Dropout on read heads")
	parser.add_argument('--read-activation',			type=str, default="tanh_abs", choices=ACTIVATION_FNS.keys())
	parser.add_argument('--disable-read-question-state',action='store_false', dest='use_read_question_state')
	
	parser.add_argument('--data-stack-width',         	type=int, default=1,   help="Width of stack entry")
	parser.add_argument('--data-stack-len',         	type=int, default=20,   help="Length of stack")
	
	parser.add_argument('--control-width',	           	type=int, default=None,	help="The width of control state")
	parser.add_argument('--control-heads',	           	type=int, default=4,	help="The number of control question-word attention heads")
	parser.add_argument('--control-dropout',	        type=float, default=0.0, help="Dropout on the control unit")

	parser.add_argument('--memory-width',	           	type=int, default=128,	help="The width of memory state")
	parser.add_argument('--memory-transform-layers',	type=int, default=1, 	help="How many deep layers in memory transforms")
	parser.add_argument('--memory-forget-activation',	type=str, default="sigmoid", 	choices=ACTIVATION_FNS.keys())
	# parser.add_argument('--memory-activation',			type=str, default="tanh", 		choices=ACTIVATION_FNS.keys())
	parser.add_argument('--memory-dropout',				type=float, default=0.0)


	parser.add_argument('--mp-activation',				type=str, default="selu", 		choices=ACTIVATION_FNS.keys())
	parser.add_argument('--mp-state-width', 			type=int, default=4)
	parser.add_argument('--disable-message-passing',    action='store_false', dest='use_message_passing')
	parser.add_argument('--disable-mp-gru', 			action='store_false', dest='use_mp_gru')
	parser.add_argument('--enable-mp-node-id', 			action='store_false', dest='use_mp_node_id')
	parser.add_argument('--enable-mp-right-shift', 		action='store_false', dest='use_mp_right_shift')
	parser.add_argument('--mp-read-heads',				type=int, default=1)

	parser.add_argument('--output-activation',			type=str, default="selu", choices=ACTIVATION_FNS.keys())
	parser.add_argument('--output-layers',				type=int, default=1)
	parser.add_argument('--output-width',	       		type=int, default=128,    help="The number of different possible answers (e.g. answer classes). Currently tied to vocab size since we attempt to tokenise the output.")

	parser.add_argument('--enable-position-encoding', 	action='store_true',  dest='use_position_encoding')
	parser.add_argument('--disable-control-cell', 		action='store_false', dest="use_control_cell")
	parser.add_argument('--disable-memory-cell', 		action='store_false', dest="use_memory_cell")
	parser.add_argument('--enable-question-state', 		action='store_true',  dest="use_question_state")
	parser.add_argument('--enable-lr-finder', 			action='store_true',  dest="use_lr_finder")
	parser.add_argument('--enable-lr-decay', 			action='store_true',  dest="use_lr_decay")
	parser.add_argument('--enable-curriculum', 			action='store_true',  dest="use_curriculum")


	parser.add_argument('--enable-tf-debug', 			action='store_true',  dest="use_tf_debug")
	parser.add_argument('--enable-comet', 				action='store_true',  dest="use_comet")
	parser.add_argument('--enable-floyd',	 			action='store_true',  dest="use_floyd")
	parser.add_argument('--diable-assert',	 			action='store_false',  dest="use_assert")
	parser.add_argument('--disable-summary-scalar', 	action='store_false', dest='use_summary_scalar')
	parser.add_argument('--enable-summary-image', 		action='store_true', dest='use_summary_image')
	
	args = vars(parser.parse_args(argv))

	args.update(generate_args_derivatives(args))
	
	# Global singleton var for easy access deep in the codebase (e.g. utility functions)
	# Note that this wont play well with PBT!! 
	# TODO: Remove
	global_args.clear()
	global_args.update(args)


	# Setup logging
	logging.basicConfig()
	tf.logging.set_verbosity(args["log_level"])
	logging.getLogger("mac-graph").setLevel(args["log_level"])

	loggers = [logging.getLogger(i) 
		for i in ["__main__", "pbt", "experiment", "macgraph", "util", "tensorflow"]]

	for i in loggers:
		i.handlers = []
		coloredlogs.install(logger=i, level=args["log_level"], fmt='%(levelname)s %(name)s %(message)s')

	return args


def save_args(args):
	pathlib.Path(args["model_dir"]).mkdir(parents=True, exist_ok=True)
	with tf.gfile.GFile(os.path.join(args["config_path"]), "w") as file:
		yaml.dump(args, file)
