
import numpy as np
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)

from .text_util import EOS_ID, UNK_ID
from .graph_util import *
from .util import tf_startswith


def parse_single_example(i):
	return tf.parse_single_example(
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

def reshape_example(args, i):
	return ({
		# Text input
		"src": 				i["src"],
		"src_len": 			i["src_len"],

		# Knowledge base
		"kb_nodes": 		tf.reshape(i["kb_nodes"], [-1, args["kb_node_width"]]),
		"kb_nodes_len":		i["kb_nodes_len"],
		"kb_edges": 		tf.reshape(i["kb_edges"], [-1, args["kb_edge_width"]]),
		"kb_edges_len":		i["kb_edges_len"],

		# Prediction stats
		"label":			i["label"], 
		"type_string":		i["type_string"],
		
	}, i["label"])

def input_fn(args, mode, question=None):

	# --------------------------------------------------------------------------
	# Read TFRecords
	# --------------------------------------------------------------------------

	d = tf.data.TFRecordDataset([args[f"{mode}_input_path"]])
	d = d.map(parse_single_example)

	# --------------------------------------------------------------------------
	# Layout input data
	# --------------------------------------------------------------------------

	d = d.map(lambda i: reshape_example(args,i))


	if args["limit"] is not None:
		d = d.take(args["limit"])

	if args["type_string_prefix"] is not None:
		d = d.filter(lambda features, labels: 
			tf_startswith(features["type_string"], args["type_string_prefix"]))

	d = d.shuffle(args["batch_size"]*1000)

	zero64 = tf.cast(0, tf.int64) 

	d = d.padded_batch(
		args["batch_size"],
		# The first three entries are the source and target line rows;
		# these have unknown-length vectors.	The last two entries are
		# the source and target row sizes; these are scalars.
		padded_shapes=(
			{
				"src": 				tf.TensorShape([None]),
				"src_len": 			tf.TensorShape([]), 

				"kb_nodes": 		tf.TensorShape([None, args["kb_node_width"]]),
				"kb_nodes_len": 	tf.TensorShape([]), 
				"kb_edges": 		tf.TensorShape([None, args["kb_edge_width"]]),
				"kb_edges_len": 	tf.TensorShape([]), 

				"label": 			tf.TensorShape([]), 
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
				"src_len": 			zero64, # unused

				"kb_nodes": 		tf.cast(UNK_ID, tf.int64), 
				"kb_nodes_len": 	zero64, # unused
				"kb_edges": 		tf.cast(UNK_ID, tf.int64), 
				"kb_edges_len": 	zero64, # unused

				"label":			zero64,
				"type_string": 		tf.cast("", tf.string),
			},
			zero64 # label (unused)
		)
	)

	# Add dynamic dimensions for convenience (e.g. to do shape assertions)
	d = d.map(lambda features, labels: ({
		**features, 
		"d_batch_size": tf.shape(features["src"])[0], 
		"d_src_len":    tf.shape(features["src"])[1],
	}, labels))
	
	return d



def gen_input_fn(args, mode):
	return lambda: input_fn(args, mode)




