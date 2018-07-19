
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
	parser.add_argument('--input-dir',  				type=str, default="./input_data/processed")
	parser.add_argument('--model-dir',      			type=str, default="./output/model")

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
	
	parser.add_argument('--answer-classes',	       		type=int, default=512,  help="The number of different possible answers (e.g. answer classes). Currently tied to vocab size since we attempt to tokenise the output.")
	parser.add_argument('--vocab-size',	           		type=int, default=512,  help="How many different words are in vocab")
	parser.add_argument('--embed-width',	       		type=int, default=64,   help="The width of token embeddings")
	
	parser.add_argument('--batch-size',            		type=int, default=32,   help="Number of items in a full batch")
	parser.add_argument('--kb-node-width',         		type=int, default=7,    help="Width of node entry into graph table aka the knowledge base")
	parser.add_argument('--kb-edge-width',         		type=int, default=3,    help="Width of edge entry into graph table aka the knowledge base")
	parser.add_argument('--control-width',	           	type=int, default=64,	help="The width of control state")
	parser.add_argument('--memory-width',	           	type=int, default=64,	help="The width of memory state")
	parser.add_argument('--num-input-layers',	   		type=int, default=3,    help="How many input layers are in the english encoding LSTM stack")
	parser.add_argument('--max-decode-iterations', 		type=int, default=8)
	parser.add_argument('--max-steps',             		type=int, default=100000)
		
	parser.add_argument('--max-gradient-norm',     		type=float, default=4.0)
	parser.add_argument('--learning-rate',         		type=float, default=0.001)
	parser.add_argument('--dropout',               		type=float, default=0.2)

	parser.add_argument('--disable-kb-nodes', action='store_false', dest='use_kb_nodes')
	parser.add_argument('--disable-kb-edges', action='store_false', dest='use_kb_edges')

	parser.add_argument('--dynamic-decode', action='store_true')
	parser.add_argument('--disable-control-cell', action='store_false', dest="use_control_cell")

	args = vars(parser.parse_args())

	args["modes"] = ["eval", "train", "predict"]

	for i in [*args["modes"], "all"]:
		args[i+"_input_path"] = os.path.join(args["input_dir"], i+"_input.tfrecords")

	args["vocab_path"] = os.path.join(args["input_dir"], "vocab.txt")
	args["question_types_path"] = os.path.join(args["input_dir"], "types.yaml")
	args["answer_classes_path"] = os.path.join(args["input_dir"], "answer_classes.yaml")



	return args
