
import argparse
import os.path

def get_args(extend=lambda parser:None):

	parser = argparse.ArgumentParser()
	extend(parser)

	parser.add_argument('--log-level',  type=str, default='INFO')
	parser.add_argument('--output-dir', type=str, default="./output")
	parser.add_argument('--input-dir',  type=str, default="./input_processed")

	parser.add_argument('--model-dir',      type=str, default="./output/model")
	parser.add_argument('--warm-start-dir', type=str, default=None)

	parser.add_argument('--eval-holdback',    type=float, default=0.1)
	parser.add_argument('--predict-holdback', type=float, default=0.005)

	parser.add_argument('--batch-size',            type=int, default=32,   help="Number of items in a full batch")
	parser.add_argument('--kb-width',              type=int, default=2,   help="Width of entry into graph table aka the knowledge base")
	parser.add_argument('--kb-len',                type=int, default=10,  help="Number of entries in graph table aka the knowledge base")
	parser.add_argument('--bus-width',	           type=int, default=32,  help="The width of instructions and cell memory")
	parser.add_argument('--embed-width',	       type=int, default=32,  help="The width of token embeddings")
	parser.add_argument('--vocab-size',	           type=int, default=32,  help="How many different words are in vocab")
	parser.add_argument('--num-input-layers',	   type=int, default=2,    help="How many input layers are in the english encoding LSTM stack")
	parser.add_argument('--limit', 				   type=int, default=None, help="How many rows of input data to train on")
	parser.add_argument('--answer-classes',	       type=int, default=32,  help="The number of different possible answers (e.g. answer classes). Currently tied to vocab size since we attempt to tokenise the output.")
	parser.add_argument('--max-decode-iterations', type=int, default=1)
	parser.add_argument('--max-steps',             type=int, default=None)


	parser.add_argument('--max-gradient-norm',     type=float, default=4.0)
	parser.add_argument('--learning-rate',         type=float, default=0.001)
	parser.add_argument('--dropout',               type=float, default=0.2)

	args = vars(parser.parse_args())

	args["modes"] = ["eval", "train", "predict"]

	for i in [*args["modes"], "all"]:
		args[i+"_input_path"] = os.path.join(args["input_dir"], i+"_input.tfrecords")

	args["vocab_path"] = os.path.join(args["input_dir"], "vocab.txt")


	return args
