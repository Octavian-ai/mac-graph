
import tensorflow as tf

from ..util import assert_shape

def control_cell(args, in_control_state, in_question_state, in_question_tokens):

	# time_minor format
	# in_question_tokens.shape = [batch_size, seq_len, bus_width]

	assert in_question_tokens.shape[-1] == args["bus_width"], f"Expected question tokens inner dimension to equal {args['bus_width']}, but got {in_question_tokens.shape[-1]} (overall shape: {in_question_tokens.shape})"
	assert len(in_question_tokens.shape) == 3

	# Skipping tf.dense(in_question_state, name="control_question_t"+iteration_step)
	all_input = tf.concat([in_control_state, in_question_state], -1)

	question_token_cmp = tf.layers.dense(all_input, args["bus_width"])
	assert_shape(question_token_cmp, [args["bus_width"]])

	question_token_dot = question_token_cmp * in_question_tokens
	assert_shape(question_token_dot, in_question_tokens.shape[1:])

	question_token_scores = tf.layers.dense(question_token_cmp, 1)
	question_token_scores = tf.nn.softmax(question_token_scores)
	
	# Expect question_token_scores.shape = [batch_size, seq_len]
	assert_shape(question_token_scores, in_question_tokens.shape[1:2]) 

	
	control_out = tf.tensordot(question_token_scores, in_question_tokens, axes=[[0], [1]])
	control_out = tf.squeeze(control_out, axis=[0])
	
	assert_shape(control_out, [args["bus_width"]])

	return control_out

