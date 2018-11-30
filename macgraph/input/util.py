
import yaml
import tensorflow as tf
import random
from tqdm import tqdm
from collections import Counter
from contextlib import ExitStack

import logging
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Miscel
# --------------------------------------------------------------------------

def min_none(a, b):
	if a is None:
		return b
	if b is None:
		return a
	return min(a,b)


# --------------------------------------------------------------------------
# TFRecord functions
# --------------------------------------------------------------------------

# Why it's so awkward to write a record I do not know

def write_int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_int64_array_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value)),

def write_boolean_array_feature(value):
	return write_int64_array_feature(value)

def write_string_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value)]))


# TODO: Better naming / structure

def parse_feature_int_array():
	return tf.FixedLenSequenceFeature([],tf.int64, allow_missing=True)

def parse_feature_boolean_array():
	return parse_feature_int_array()

def parse_feature_string():
	return tf.FixedLenSequenceFeature([],tf.string, allow_missing=True)

def parse_feature_int():
	return tf.FixedLenFeature([], tf.int64)


# --------------------------------------------------------------------------
# TF helpers
# --------------------------------------------------------------------------

def tf_startswith(tensor, prefix, axis=None):
	return tf.reduce_all(tf.equal(tf.substr(tensor, 0, len(prefix)), prefix), axis=axis)



# --------------------------------------------------------------------------
# File readers and writers
# --------------------------------------------------------------------------

def read_gqa(args, limit=None):

	if limit is None:
		limit = args["limit"]

	with ExitStack() as stack:
		files = [stack.enter_context(open(fname)) for fname in args["gqa_paths"]]
		
		in_files = [
			stack.enter_context(tf.gfile.GFile(i, 'r'))
			for i in args["gqa_paths"]
		]

		yamls = [
			yaml.safe_load_all(i)
			for i in in_files
		]

		ctr = 0

		for row in zip(*yamls):
			for i in row:
				if i is not None:
					if args["filter_type_prefix"] is None or i["question"]["type_string"].startswith(args["filter_type_prefix"]):
						yield i
						ctr += 1
						if limit is not None and ctr >= limit:
							logger.debug("Hit limit, stop")
							return
					else:
						logger.debug(f"{i['question']['type_string']} does not match prefix {args['filter_type_prefix']}")
				else:
					logger.debug("Skipping None yaml doc")




# --------------------------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------------------------

def StringDataset(s):

	def generator():
		yield s

	return tf.data.Dataset.from_generator(generator, tf.string, tf.TensorShape([]) )





			