
import tensorflow as tf
import numpy as np

from .model import model_fn
from .input import gen_input_fn
from .args import get_args

from .input import Vocab

def predict(args):
	estimator = tf.estimator.Estimator(model_fn, model_dir=args["model_dir"], params=args)
	predictions = estimator.predict(input_fn=gen_input_fn(args, "predict"))

	vocab = Vocab.load(args)

	for p in predictions:
		for k, v in p.items():
			try:
				s = vocab.ids_to_string(v)
			except:
				s = vocab.inverse_lookup(v)
			
			print(f"{k}: {s}")

		print("-------")


if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.DEBUG)
	args = get_args()
	predict(args)



