
import tensorflow as tf

from ..util import *
from ..attention import *

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

		question_token_query = tf.layers.dense(all_input, args["bus_width"], activation=tf.nn.tanh)
		question_token_query = tf.layers.dense(question_token_query, args["bus_width"], activation=tf.nn.tanh)
		question_token_query = dynamic_assert_shape(question_token_query, 
			[ features["d_batch_size"], args["bus_width"] ]
		)

		control_out = attention(args, question_token_query, in_question_tokens)

		# Hack to let the question state through if wanted
		control_out = tf.layers.dense(
			tf.concat([
				control_out, 
				tf.layers.dense(in_question_state, args["bus_width"], activation=tf.nn.tanh)
			], -1)
		, args["bus_width"])



		return control_out

