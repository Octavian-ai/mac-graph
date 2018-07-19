
import tensorflow as tf

from ..util import *
from ..attention import *

def control_cell(args, features, in_control_state, in_question_state, in_question_tokens):
	"""
	Build a control cell

	Data in time-minor format

	Arguments:
		- in_control_state.shape   = [batch_size, control_width]
		- in_question_state.shape  = [batch_size, embed_width]
		- in_question_tokens.shape = [batch_size, seq_len, embed_width]

	"""
	with tf.name_scope("control_cell"):

		in_control_state = dynamic_assert_shape(in_control_state, 
			[ features["d_batch_size"], args["control_width"] ]
		)
		
		# Skipping tf.dense(in_question_state, name="control_question_t"+iteration_step)
		all_input = tf.concat([in_control_state, in_question_state], -1, name="all_input")

		question_token_query = deeep(all_input, args["embed_width"])
		question_token_query = dynamic_assert_shape(question_token_query, 
			[ features["d_batch_size"], args["embed_width"] ]
		)

		control_out = attention(in_question_tokens, question_token_query)
		# control_out = tf.layers.dense(control_out, args["control_width"], name="resize_control_out")
		
		return control_out

