
import argparse
import os

def get_args(args=None):

	parser = argparse.ArgumentParser()

	parser.add_argument('--run',					type=str,  default=os.getenv("RUN", "default"), help="Prefix used for file storage and messaging")
	parser.add_argument('--log-level',  			type=str, default='INFO')

	parser.add_argument('--output-dir', 			type=str, default="./output/cluster/")
	parser.add_argument('--input-dir',  			type=str, default="./input_data/processed")
	parser.add_argument('--model-dir',  		    type=str, default="./output/cluster/model")

	parser.add_argument('--queue-type',				type=str,  default="rabbitmq", choices=["rabbitmq","google"])
	parser.add_argument('--amqp-url',				type=str,  default=os.getenv("AMQP_URL", 'amqp://guest:guest@localhost'))

	

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
	
	return parser.parse_args(args)
