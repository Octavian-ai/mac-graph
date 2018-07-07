
import tensorflow as tf

from ..util import *

def control_cell(args, features, in_control_state, in_question_state, in_question_tokens):
	"""
	Build a control cell

	Data in time-minor format

	Arguments:
		- in_control_state.shape   = [batch_size, bus_width]
		- in_question_state.shape  = [batch_size, bus_width]
		- in_question_tokens.shape = [batch_size, seq_len, bus_width]

	"""
	with tf.name_scope("control_cell"):

		in_control_state = dynamic_assert_shape(in_control_state, 
			[ features["d_batch_size"], args["bus_width"] ]
		)
		
		# Skipping tf.dense(in_question_state, name="control_question_t"+iteration_step)
		all_input = tf.concat([in_control_state, in_question_state], -1, name="all_input")

		question_token_cmp = tf.layers.dense(all_input, args["bus_width"])
		question_token_cmp = dynamic_assert_shape(question_token_cmp, 
			[ features["d_batch_size"], args["bus_width"] ]
		)

		question_token_dot = tf.expand_dims(question_token_cmp, 1) * in_question_tokens
		question_token_dot = dynamic_assert_shape(question_token_dot, 
			[ features["d_batch_size"], features["d_seq_len"], args["bus_width"] ]
		)
	
		question_token_scores = tf.layers.dense(question_token_dot, 1, name="question_token_scores")
		question_token_scores = tf.squeeze(question_token_scores, axis=-1)
		question_token_scores = tf.nn.softmax(question_token_scores)
		question_token_scores = dynamic_assert_shape(question_token_scores,
			[ features["d_batch_size"], features["d_seq_len"] ]
		)

		control_out = tf.tensordot(question_token_scores, in_question_tokens, axes=[[1], [1]])
		control_out = tf.squeeze(control_out, axis=[0])
		control_out = dynamic_assert_shape(control_out, 
			[ features["d_batch_size"], args["bus_width"]]
		)

		return control_out

