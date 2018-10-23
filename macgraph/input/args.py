
from ..args import get_args as get_args_parent

def get_args(extend=lambda x:None, argv=None):
	def inner_extend(parser):
		parser.add_argument('--skip-vocab', 		action='store_true')
		parser.add_argument('--gqa-dir', 			type=str, default="./input_data/raw")
		parser.add_argument('--gqa-path', 			type=str, default=None)
		parser.add_argument('--balance-batch', 		type=int, default=1000)
		parser.add_argument('--vocab-build-limit', 	type=int, default=2000, help="It's slow to read all records to build vocab, so just read this many. Could cause UNK to slip into dataset depending on distribution of tokens.")

		extend(parser)

	return get_args_parent(inner_extend, argv)