
import argparse
import os.path

def get_args():

	parser = argparse.ArgumentParser()

	parser.add_argument('--batch-size',            type=int, default=16,  help="Number of items in a full batch")
	parser.add_argument('--kb-width',              type=int, default=16,  help="Width of entry into graph table aka the knowledge base")
	parser.add_argument('--kb-len',                type=int, default=128, help="Number of entries in graph table aka the knowledge base")
	parser.add_argument('--bus-width',	           type=int, default=32,  help="The width of instructions and cell memory")
	parser.add_argument('--vocab-size',	           type=int, default=60,  help="How many different words are in vocab")
	parser.add_argument('--num-input-layers',	   type=int, default=2,   help="How many input layers are in the english encoding LSTM stack")
	parser.add_argument('--max-decode-iterations', type=int, default=32)
	parser.add_argument('--answer-classes',	       type=int, default=32,  help="The number of different possible answers (e.g. answer classes")
	
	parser.add_argument('--learning-rate', type=float, default=0.001)
	parser.add_argument('--dropout', type=float, default=0.2)

	return vars(parser.parse_args())
