
import tensorflow as tf

from ..util import assert_shape

def control_cell(args, in_control_state, in_question_state, in_question_tokens):

	# time_minor format
	# in_question_tokens.shape = [batch_size, seq_len, num_units]

	assert in_question_tokens.shape[-1] == args["question_num_units"]
	assert len(in_question_tokens.shape) == 3

	# Skipping tf.dense(in_question_state, name="control_question_t"+iteration_step)
	all_input = tf.concat([in_control_state, in_question_state], -1)

	question_word_cmp = tf.layer.dense(all_input, [args["question_num_units"]])
	assert_shape(question_word_cmp, [args["question_num_units"]])

	question_word_dot = question_word_cmp * in_question_tokens
	assert_shape(question_word_dot, in_question_tokens.shape[1:3])

	question_scores = tf.layer.dense(question_word_cmp, 1)
	question_scores = tf.nn.softmax(question_scores)
	# Expect question_scores.shape = [batch_size, seq_len]
	assert_shape(question_scores, in_question_tokens.shape[1:2]) 

	control_out = tf.tensordot(question_scores, in_question_tokens, axes=[[1], [1]])

	return control_out

