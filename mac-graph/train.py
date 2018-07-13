
import tensorflow as tf

from .model import model_fn
from .input import gen_input_fn
from .args import get_args
from .predict import predict

def train(args):

	estimator = tf.estimator.Estimator(model_fn, 
		model_dir=args["model_dir"],
		warm_start_from=args["warm_start_dir"],
		params=args)

	train_spec = tf.estimator.TrainSpec(input_fn=gen_input_fn(args, "train"), max_steps=args["max_steps"])
	eval_spec  = tf.estimator.EvalSpec(input_fn=gen_input_fn(args, "eval"))

	tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)



if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.DEBUG)
	args = get_args()

	train_size = sum(1 for _ in tf.python_io.tf_record_iterator(args["train_input_path"]))
	tf.logging.debug(f"Training on {train_size} records")

	train(args)
	predict(args)



