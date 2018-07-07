
import tensorflow as tf

from ..util import *


def read_from_graph(args, in_content, in_mask, in_knowledge, W_score=None):
	"""Perform attention based read from table

	@param W_score is for testing/debug purposes so you can easily inject the score fn you'd like. The code will default to a variable normally
	
	@returns read_data
	"""

	assert_shape(in_content, [args["kb_width"]])
	assert_shape(in_mask, [args["kb_width"]])
	assert_shape(in_knowledge, [args["kb_len"], args["kb_width"]])

	assert in_mask.dtype == tf.float32, "Mask should be floats between 0 and 1"

	with tf.variable_scope("read_from_graph", reuse=tf.AUTO_REUSE):

		if W_score is None:
			W_score = tf.get_variable("W_score", [args["kb_width"], args["kb_width"]], tf.float32)
		
		masked_query = in_content * in_mask
		assert_shape(masked_query, [args["kb_width"]])

		masked_kb = in_knowledge * tf.expand_dims(in_mask, 1)
		assert_shape(masked_kb, [args["kb_len"], args["kb_width"]])

		# --------------------------------------------------------------------------
		# Perform scoring (masked_query x W x masked_kb)
		# --------------------------------------------------------------------------
		
		halfbaked = tf.matmul(masked_query, W_score)
		assert_shape(halfbaked, [args["kb_width"]])
		
		# todo: verify this is right
		scores = tf.matmul(masked_kb, tf.expand_dims(halfbaked, -1), name="scores")
		scores = tf.squeeze(scores, axis=-1)
		assert_shape(scores, [args["kb_len"]])
		scores = tf.nn.softmax(scores)

		weighted_kb = in_knowledge * tf.expand_dims(scores, -1)
		assert_shape(weighted_kb, [args["kb_len"], args["kb_width"]])
		
		read_data = tf.reduce_sum(weighted_kb, axis=1)
		assert_shape(read_data, [args["kb_width"]])

		return read_data



def read_cell(args, in_memory_state, in_control, in_knowledge, W_score=None):
	"""
	A read cell

	@returns read_data

	"""

	assert_shape(in_memory_state,   [args["bus_width"]])
	assert_shape(in_control, [args["bus_width"]])

	in_all = tf.concat([in_memory_state, in_control], -1)

	read_content = tf.layers.dense(in_all, args["kb_width"])
	read_mask    = tf.layers.dense(in_all, args["kb_width"])

	read_data = read_from_graph(args, read_content, read_mask, in_knowledge, W_score)
	assert_shape(read_data, [args["bus_width"]])

	return read_data




