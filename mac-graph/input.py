
import tensorflow as tf

def gen_input_fn(args, mode):

	return lambda:tf.data.Dataset.from_generator()