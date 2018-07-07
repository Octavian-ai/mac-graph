
import tensorflow as tf

from ..util import *

def control_cell(args, features, in_control_state, in_question_state, in_question_tokens):
	with tf.name_scope("control_cell"):

		assert_shape(in_control_state, [args["bus_width"]])

		# time_minor format
		# in_question_tokens.shape = [batch_size, seq_len, bus_width]
		in_question_tokens = dynamic_assert_shape(in_question_tokens, 
			[ features["d_batch_size"], features["d_seq_len"], args["bus_width"] ]
		)

		in_question_state = dynamic_assert_shape(in_question_state,
			[ features["d_batch_size"], args["bus_width"] ]
		)

		# Skipping tf.dense(in_question_state, name="control_question_t"+iteration_step)
		# Concatenating the first dimension seems wrong as it screws with the batch
		all_input = tf.concat([in_control_state, in_question_state], 0, name="all_input")

		question_token_cmp = tf.layers.dense(all_input, args["bus_width"])

		question_token_cmp = dynamic_assert_shape(question_token_cmp, 
			[ features["d_batch_size"], args["bus_width"] ]
		)

		question_token_dot = question_token_cmp * in_question_tokens

		# question_token_dot.shape = [batch_size, seq_len, bus_width]

		question_token_scores = tf.layers.dense(question_token_dot, 1, name="question_token_scores")
		question_token_scores = tf.squeeze(question_token_scores, axis=-1)
		question_token_scores = tf.nn.softmax(question_token_scores)
		
		# Expect question_token_scores.shape = [batch_size, seq_len]
		assert_shape(question_token_scores, [in_question_tokens.shape[1]])

		control_out = tf.tensordot(question_token_scores, in_question_tokens, axes=[[1], [1]])
		control_out = tf.squeeze(control_out, axis=[0])
		
		assert_shape(control_out, [args["bus_width"]])

		return control_out

