
import yaml
import tensorflow as tf
import random
from tqdm import tqdm
from collections import Counter

import logging
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# TFRecord functions
# --------------------------------------------------------------------------

# Why it's so awkward to write a record I do not know

def int32_feature(value):
	return tf.train.Feature(int32_list=tf.train.Int32List(value=[value]))

def int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def string_feature(value):
	return conv_bytes_feature(value)

def conv_bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value)]))


# --------------------------------------------------------------------------
# TF helpers
# --------------------------------------------------------------------------

def tf_startswith(tensor, prefix, axis=None):
	return tf.reduce_all(tf.equal(tensor[:len(prefix)], prefix), axis=axis)

# --------------------------------------------------------------------------
# File readers and writers
# --------------------------------------------------------------------------

def read_gqa(args):
	with tf.gfile.GFile(args["gqa_path"], 'r') as in_file:
		d = yaml.safe_load_all(in_file)

		ctr = 0

		for i in d:
			if i is not None:
				if args["type_string_prefix"] is None or i["question"]["type_string"].startswith(args["type_string_prefix"]):
					yield i
					ctr += 1
					if args["limit"] is not None and ctr >= args["limit"]:
						logger.debug("Hit limit, stop")
						return
				else:
					logger.debug(f"{i['question']['type_string']} does not match prefix {args['type_string_prefix']}")
			else:
				logger.debug("Skipping None yaml doc")

class Partitioner(object):

	def __init__(self, args):
		self.args = args
		self.written = 0


	def __enter__(self, *vargs):
		self.files = {
			i: tf.python_io.TFRecordWriter(self.args[f"{i}_input_path"]) 
			for i in self.args['modes']
		}

		return self


	def write(self, *vargs):
		r = random.random()

		if r < self.args["eval_holdback"]:
			mode = "eval"
		elif r < self.args["eval_holdback"] + self.args["predict_holdback"]:
			mode = "predict"
		else:
			mode = "train"

		self.files[mode].write(*vargs)
		self.written += 1


	def __exit__(self, *vargs):
		for i in self.files.values():
			i.close()

		self.files = None


class Balancer(object):
	"""Streaming oversampler. Will only oversample classes seen in batch, bigger frequency is better"""

	def __init__(self, partitioner, hold_back=100):
		self.partitioner = partitioner
		self.total_classes = Counter()
		self.records_by_class = {}
		self.batch_i = 0
		self.hold_back = hold_back

	def __enter__(self, *vargs):
		return self

	def __exit__(self, *vargs):
		logger.debug(f"Classes after oversampling: {self.total_classes}")
		self.oversample()

	# --------------------------------------------------------------------------
	# Class balancing
	# --------------------------------------------------------------------------

	def record_batch_item(self, doc, record):
		self.total_classes[doc["answer"]] += 1

		if doc["answer"] not in self.records_by_class:
			self.records_by_class[doc["answer"]] = []

		self.records_by_class[doc["answer"]].append(record)
		if len(self.records_by_class[doc["answer"]]) > self.hold_back:
			self.records_by_class[doc["answer"]] = self.records_by_class[doc["answer"]][-self.hold_back:]
		
		assert len(self.records_by_class[doc["answer"]]) <= self.hold_back

		self.batch_i += 1

	def oversample(self):
		if len(self.total_classes) > 0:
			target = max(self.total_classes.values())

			for key, count in self.total_classes.items():
				if count < target:
					delta = target - count
					logger.debug(f"Oversampling {key} x {delta}")
					for i in range(delta):
						self.partitioner.write(random.choice(self.records_by_class[key]))
						self.total_classes[key] += 1
						
			self.batch_classes = Counter()
			self.batch_i = 0

	def oversample_every(self, freq):
		if self.batch_i >= freq:
			self.oversample()



# --------------------------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------------------------

def StringDataset(s):

	def generator():
		yield s

	return tf.data.Dataset.from_generator(generator, tf.string, tf.TensorShape([]) )





			