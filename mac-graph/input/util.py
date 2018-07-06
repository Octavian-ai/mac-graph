

import tensorflow as tf


# --------------------------------------------------------------------------
# TFRecord functions
# --------------------------------------------------------------------------

# Why it's so awkward to write a record I do not know

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _conv_bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value)]))




# --------------------------------------------------------------------------
# File readers and writers
# --------------------------------------------------------------------------

def read_gqa(args):
	with tf.gfile.GFile(args["gqa_path"], 'r') as in_file:
		d = yaml.safe_load_all(in_file)

		for i in d:
			yield i


class Partitioner(object):

	def __init__(self, args):
		self.args = args


	def __enter__(self, *vargs):
		self.files = {i: tf.python_io.TFRecordWriter(args[f"{i}_input_path"], "w") for i in self.args.modes}
		return self


	def write(self, *vargs):
		r = random.random()

		if r < args["eval_holdback"]:
			mode = "eval"
		elif r < args["eval_holdback"] + args["predict_holdback"]:
			mode = "predict"
		else:
			mode = "train"

		self.files[r].write(*vargs)


	def __exit__(self, *vargs):
		for i in self.files.values():
			i.close()


# --------------------------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------------------------

def StringDataset(s):

	def generator():
		yield s

	return tf.data.Dataset.from_generator(generator, tf.string, tf.TensorShape([]) )





			