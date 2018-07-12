
import numpy as np
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)

from .text_util import EOS_ID
from .graph_util import *


def input_fn(args, mode, question=None):

	d = tf.data.TFRecordDataset([args[f"{mode}_input_path"]])

	# Parse
	d = d.map(lambda i: tf.parse_single_example(
		i,
		features = {
			'src': 				tf.FixedLenSequenceFeature([],tf.int64, allow_missing=True),
			'src_len': 			tf.FixedLenFeature([], tf.int64),
			'kb_edges': 		tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
			'kb_edges_len': 	tf.FixedLenFeature([], tf.int64),
			'kb_nodes': 		tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
			'kb_nodes_len': 	tf.FixedLenFeature([], tf.int64),
			'label': 			tf.FixedLenFeature([], tf.int64),
			'type_string':		tf.FixedLenSequenceFeature([], tf.string, allow_missing=True),
		})
	)

	def as_2D_shape(f):
		return tf.concat([
			[tf.constant(-1, dtype=tf.int64)],
			[f]
		], 0)

	d = d.map(lambda i: ({
		"src": 				i["src"],
		"src_len": 			i["src_len"],
		"label":			i["label"], # For prediction comparion
		"knowledge_base": 	tf.reshape(i["kb_nodes"], [-1, args["kb_width"]]), # as_2D_shape(i["kb_width"]),
		"type_string":		i["type_string"], # For prediction stats
	}, i["label"]))


	if args["limit"] is not None:
		d = d.take(args["limit"])

	d = d.shuffle(args["batch_size"]*10)

	d = d.padded_batch(
		args["batch_size"],
		# The first three entries are the source and target line rows;
		# these have unknown-length vectors.	The last two entries are
		# the source and target row sizes; these are scalars.
		padded_shapes=(
			{
				"src": 				tf.TensorShape([None]),
				"src_len": 			tf.TensorShape([]), 
				"label": 			tf.TensorShape([]), 
				"knowledge_base": 	tf.TensorShape([None, args["kb_width"]]),
				"type_string": 		tf.TensorShape([None]),
			},
			tf.TensorShape([]),	# label
		),
			
		# Pad the source and target sequences with eos tokens.
		# (Though notice we don't generally need to do this since
		# later on we will be masking out calculations past the true sequence.
		padding_values=(
			{
				"src": 				tf.cast(EOS_ID, tf.int64), 
				"src_len": 			tf.cast(0, tf.int64), # unused
				"label": 			tf.cast(0, tf.int64), # unused
				"knowledge_base": 	tf.cast(0, tf.int64), # unused
				"type_string": 		tf.cast("", tf.string), # unused
			},
			tf.cast(0, tf.int64) # label (unused)
		)
	)

	# Add dynamic dimensions for convenience (e.g. shape assertions)
	d = d.map(lambda features, labels: ({
		**features, 
		"d_batch_size": tf.shape(features["src"])[0], 
		"d_seq_len": tf.shape(features["src"])[1],
	}, labels))
	
	return d



def gen_input_fn(args, mode):
	return lambda: input_fn(args, mode)




