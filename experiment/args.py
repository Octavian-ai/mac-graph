
import argparse
import os

def get_args(args=None):

	parser = argparse.ArgumentParser()

	parser.add_argument('--run',					type=str,  default=os.getenv("RUN", "default"), help="Prefix used for file storage and messaging")

	parser.add_argument('--log-level',  type=str, default='INFO')
	parser.add_argument('--output-dir', type=str, default="./output")
	parser.add_argument('--input-dir',  type=str, default="./input_data/processed")

	parser.add_argument('--model-dir',      type=str, default="./output/model")
	parser.add_argument('--warm-start-dir', type=str, default=None)

	parser.add_argument('--eval-holdback',    type=float, default=0.1)
	parser.add_argument('--predict-holdback', type=float, default=0.005)

	parser.add_argument('--batch-size',            type=int, default=32,  help="Number of items in a full batch")
	parser.add_argument('--kb-node-width',         type=int, default=7,   help="Width of node entry into graph table aka the knowledge base")
	parser.add_argument('--kb-edge-width',         type=int, default=3,   help="Width of edge entry into graph table aka the knowledge base")
	parser.add_argument('--bus-width',	           type=int, default=64,  help="The width of instructions and cell memory")
	parser.add_argument('--embed-width',	       type=int, default=64,  help="The width of token embeddings")
	parser.add_argument('--vocab-size',	           type=int, default=64,  help="How many different words are in vocab")
	parser.add_argument('--num-input-layers',	   type=int, default=3,   help="How many input layers are in the english encoding LSTM stack")
	parser.add_argument('--limit', 				   type=int, default=None,help="How many rows of input data to train on")
	parser.add_argument('--answer-classes',	       type=int, default=32,  help="The number of different possible answers (e.g. answer classes). Currently tied to vocab size since we attempt to tokenise the output.")
	parser.add_argument('--max-decode-iterations', type=int, default=8)
	parser.add_argument('--max-steps',             type=int, default=100000)

	parser.add_argument('--max-gradient-norm',     type=float, default=4.0)
	parser.add_argument('--learning-rate',         type=float, default=0.001)
	parser.add_argument('--dropout',               type=float, default=0.2)

	# For storing to Google Cloud
	parser.add_argument('--bucket',					type=str,  default=None)
	parser.add_argument('--gcs-dir',				type=str,  default=None)
	parser.add_argument('--project',				type=str,  default=os.getenv("GOOGLE_CLOUD_PROJECT", "octavian-181621"))

	parser.add_argument('--epochs', 				type=int,  default=1000)
	parser.add_argument('--micro-step', 			type=int,  default=os.getenv("MICRO_STEP", 1000))
	parser.add_argument('--macro-step', 			type=int,  default=os.getenv("MACRO_STEP", 10))

	parser.add_argument('--n-workers', 				type=int,  default=os.getenv("N_WORKERS", 15))
	parser.add_argument('--n-drones', 				type=int,  default=os.getenv("N_DRONES", 1))
	parser.add_argument('--job-timeout', 			type=int,  default=3*60)
	parser.add_argument('--message-timeout', 		type=int,  default=60)
	parser.add_argument('--sleep-per-cycle', 		type=int,  default=5)
	parser.add_argument('--save-secs', 				type=int,  default=30)
	parser.add_argument('--print-secs', 			type=int,  default=60)
	
	parser.add_argument('--heat',					type=float, default=1.0)
	parser.add_argument('--exploit-pct',			type=float, default=0.2, help="The % to cull, and to reproduce from")
	parser.add_argument('--sexual-compatibility-min',type=float, default=0.5, help="Minimum similarity for sexual reproduction (e.g. % place in stack rank")
	parser.add_argument('--sexual-compatibility-max',type=float, default=0.8)
	parser.add_argument('--sexual-top-candidates',  type=int,   default=4, help="How many of the top candidates to randomly choose from")



	parser.add_argument('--profile',				action='store_true')
	parser.add_argument('--single-threaded',		action='store_true')
	parser.add_argument('--log-tf',					action='store_true')
	parser.add_argument('--floyd-metrics',			action='store_true')
	parser.add_argument('--breed-sexual',			action='store_true')



	parser.add_argument('--disable-save',			action='store_false',dest="save")
	parser.add_argument('--disable-load',			action='store_false',dest="load")
	parser.add_argument('--master-works', 			action='store_true',help="Master will also act as drone")
	parser.add_argument('--run-baseline', 			action='store_true',help="Run static baseline tests")
	
	
	parser.add_argument('--queue-type',				type=str,  default="rabbitmq", choices=["rabbitmq","google"])
	parser.add_argument('--amqp-url',				type=str,  default=os.getenv("AMQP_URL", 'amqp://guest:guest@172.17.0.2:5672/ashwath'))


	args = vars(parser.parse_args())

	args["modes"] = ["eval", "train", "predict"]

	for i in [*args["modes"], "all"]:
		args[i+"_input_path"] = os.path.join(args["input_dir"], i+"_input.tfrecords")

	args["vocab_path"] = os.path.join(args["input_dir"], "vocab.txt")
	args["types_path"] = os.path.join(args["input_dir"], "types.yaml")

	return parser.parse_args(args)
