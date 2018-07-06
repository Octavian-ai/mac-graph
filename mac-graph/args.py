
import argparse
import os.path

def get_args():

	parser = argparse.ArgumentParser()

	parser.add_argument('--batch-size',         type=int, default=16,  help="Number of items in a full batch")
	parser.add_argument('--kb-width',           type=int, default=16,  help="Width of entry into graph table aka the knowledge base")
	parser.add_argument('--kb-len',             type=int, default=128, help="Number of entries in graph table aka the knowledge base")
	parser.add_argument('--bus-width',	        type=int, default=32,  help="The width of instructions and cell memory")
	parser.add_argument('--question-num-units',	type=int, default=32,  help="The number of units in the question pre-processing RNN")

	return vars(parser.parse_args())
