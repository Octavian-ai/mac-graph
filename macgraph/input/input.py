
import numpy as np
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)

from .text_util import EOS_ID, UNK_ID
from .graph_util import *
from .util import *

def parse_single_example(i):
	return tf.parse_single_example(
		i,
		features = {
			'src': 				parse_feature_int_array(),
			'src_len': 			parse_feature_int(),
			
			'kb_edges': 		parse_feature_int_array(),
			'kb_edges_len': 	parse_feature_int(),
			'kb_nodes': 		parse_feature_int_array(),
			'kb_nodes_len': 	parse_feature_int(),
			'kb_adjacency':		parse_feature_boolean_array(),
			
			'label': 			parse_feature_int(),
			'type_string':		parse_feature_string(),
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
		"kb_adjacency":		tf.reshape(i["kb_adjacency"], [args["kb_node_max_len"], args["kb_node_max_len"]]),

		# Prediction stats
		"label":			i["label"], 
		"type_string":		i["type_string"],
		
	}, i["label"])

def switch_to_from(db):
	return tf.stack([db[:,2], db[:,1], db[:,0]], -1)

def make_edges_bidirectional(features, labels):
	features["kb_edges"] = tf.concat([features["kb_edges"], switch_to_from(features["kb_edges"])], 0)
	features["kb_edges_len"] *= 2
	return features, labels


def reshape_adjacency(features, labels):
	a = features["kb_adjacency"]

	nsq = features["kb_nodes_len"] * features["kb_nodes_len"]
	a = a[:, 0: ]
	a = tf.reshape(a, [
		features["d_batch_size"], 
		features["kb_nodes_len"], 
		features["kb_nodes_len"]  
	])

	features["kb_adjacency"] = a

	return features, labels

def cast_adjacency_to_bool(features, labels):
	features["kb_adjacency"] = tf.cast(features["kb_adjacency"], tf.bool)
	return features, labels

def input_fn(args, mode, question=None, repeat=True):

	# --------------------------------------------------------------------------
	# Read TFRecords
	# --------------------------------------------------------------------------

	d = tf.data.TFRecordDataset([args[f"{mode}_input_path"]])
	d = d.map(parse_single_example)

	# --------------------------------------------------------------------------
	# Layout input data
	# --------------------------------------------------------------------------

	d = d.map(lambda i: reshape_example(args,i))
	d = d.map(make_edges_bidirectional)


	if args["limit"] is not None:
		d = d.take(args["limit"])

	if args["filter_type_prefix"] is not None:
		d = d.filter(lambda features, labels: 
			tf_startswith(features["type_string"], args["filter_type_prefix"]))

	# if args["filter_output_class"] is not None:
	# 	classes_as_ints = [args["vocab"].inverse_lookup(i) for i in args["filter_output_class"]]
	# 	d = d.filter(lambda features, labels: 
	# 		tf.reduce_any(tf.equal(features["label"], classes_as_ints))
	# 	)

	d = d.shuffle(args["batch_size"]*1000)

	zero_64 = tf.cast(0, tf.int64) 
	unk_64  = tf.cast(UNK_ID, tf.int64)

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
				"kb_adjacency": 	tf.TensorShape([args["kb_node_max_len"], args["kb_node_max_len"]]),

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
				"src_len": 			zero_64, # unused

				"kb_nodes": 		unk_64, 
				"kb_nodes_len": 	zero_64, # unused
				"kb_edges": 		unk_64, 
				"kb_edges_len": 	zero_64, # unused
				"kb_adjacency": 	zero_64, 

				"label":			zero_64,
				"type_string": 		tf.cast("", tf.string),
			},
			zero_64 # label (unused)
		),
		drop_remainder=(mode == "predict")
	)

	d = d.map(cast_adjacency_to_bool)

	# Add dynamic dimensions for convenience (e.g. to do shape assertions)
	d = d.map(lambda features, labels: ({
		**features, 
		"d_batch_size": tf.shape(features["src"])[0], 
		"d_src_len":    tf.shape(features["src"])[1],
	}, labels))

	if repeat:
		d = d.repeat()
	
	return d



def gen_input_fn(args, mode):
	return lambda: input_fn(args, mode)




