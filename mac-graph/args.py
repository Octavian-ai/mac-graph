
import argparse
import os.path

def get_args(extend=lambda parser:None):

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
		
	parser.add_argument('--max-gradient-norm',     		type=float, default=4.0)
	parser.add_argument('--learning-rate',         		type=float, default=0.001)
	parser.add_argument('--dropout',               		type=float, default=0.2)

	# --------------------------------------------------------------------------
	# Network topology
	# --------------------------------------------------------------------------

	parser.add_argument('--answer-classes',	       		type=int, default=8,    help="The number of different possible answers (e.g. answer classes). Currently tied to vocab size since we attempt to tokenise the output.")
	parser.add_argument('--vocab-size',	           		type=int, default=90,   help="How many different words are in vocab")
	parser.add_argument('--embed-width',	       		type=int, default=32,   help="The width of token embeddings")
	parser.add_argument('--pos-enc-width',	       		type=int, default=0,   help="The width of token embeddings")
	parser.add_argument('--num-input-layers',	   		type=int, default=3,    help="How many input layers are in the english encoding LSTM stack")
	parser.add_argument('--max-seq-len',	  	 		type=int, default=20,   help="Maximum length of question token list")


	parser.add_argument('--kb-node-width',         		type=int, default=7,    help="Width of node entry into graph table aka the knowledge base")
	parser.add_argument('--kb-node-max-len',         	type=int, default=40,   help="Width of node entry into graph table aka the knowledge base")
	parser.add_argument('--kb-edge-width',         		type=int, default=3,    help="Width of edge entry into graph table aka the knowledge base")
	parser.add_argument('--kb-edge-max-len',         	type=int, default=40,    help="Width of edge entry into graph table aka the knowledge base")
	
	parser.add_argument('--read-heads',         		type=int, default=1,    help="Number of read heads for each knowledge base tabel")
	
	parser.add_argument('--data-stack-width',         	type=int, default=64,   help="Width of stack entry")
	parser.add_argument('--data-stack-len',         	type=int, default=20,   help="Length of stack")
	parser.add_argument('--control-width',	           	type=int, default=64,	help="The width of control state")
	parser.add_argument('--control-heads',	           	type=int, default=1,	help="The number of control question-word attention heads")
	
	parser.add_argument('--memory-width',	           	type=int, default=128,	help="The width of memory state")
	parser.add_argument('--memory-transform-layers',	type=int, default=2, 	help="How many deep layers in memory transforms")

	parser.add_argument('--disable-kb-node', 			action='store_false', dest='use_kb_node')
	parser.add_argument('--disable-kb-edge', 			action='store_false', dest='use_kb_edge')
	parser.add_argument('--enable-data-stack', 			action='store_true',  dest='use_data_stack')
	parser.add_argument('--disable-control-cell', 		action='store_false', dest="use_control_cell")
	parser.add_argument('--disable-dynamic-decode', 	action='store_false', dest="use_dynamic_decode")
	parser.add_argument('--max-decode-iterations', 		type=int, default=2)
	

	args = vars(parser.parse_args())

	args["modes"] = ["eval", "train", "predict"]

	for i in [*args["modes"], "all"]:
		args[i+"_input_path"] = os.path.join(args["input_dir"], i+"_input.tfrecords")

	args["vocab_path"] = os.path.join(args["input_dir"], "vocab.txt")
	args["question_types_path"] = os.path.join(args["input_dir"], "types.yaml")
	args["answer_classes_path"] = os.path.join(args["input_dir"], "answer_classes.yaml")



	return args
