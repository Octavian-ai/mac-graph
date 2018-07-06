
import tensorflow as tf

from ..util import assert_shape


def read_from_graph(args, in_content, in_mask, in_knowledge, W_score=None):
	"""Perform attention based read from table

	@param W_score is for testing/debug purposes so you can easily inject the score fn you'd like. The code will default to a variable normally
	"""

	assert_shape(in_content, [args["kb_width"]])
	assert_shape(in_mask, [args["kb_width"]])
	assert_shape(in_knowledge, [args["kb_len"], args["kb_width"]], batchless=True)

	assert in_mask.dtype == tf.float32, "Mask should be floats between 0 and 1"

	with tf.variable_scope("read_from_graph", reuse=tf.AUTO_REUSE):

		if W_score is None:
			W_score = tf.get_variable("W_score", [args["kb_width"], args["kb_width"]], tf.float32)
		
		query = in_content * in_mask
		assert_shape(query, [args["kb_width"]])
		
		halfbaked = tf.matmul(query, W_score)
		assert_shape(halfbaked, [args["kb_width"]])

		masked_kb = in_knowledge * in_mask
		
		scores = tf.tensordot(halfbaked, masked_kb, axes=[[1], [1]], name="scores")
		assert_shape(scores, [args["kb_len"]])
		scores_sm = tf.nn.softmax(scores)

		kb_linear_comb = tf.tensordot(scores_sm, in_knowledge, axes=[[1], [0]])
		assert_shape(kb_linear_comb, [args["kb_width"]])

		return kb_linear_comb



def read_cell(args, in_query, in_state, in_knowledge, out_state, out_answer):

	return True