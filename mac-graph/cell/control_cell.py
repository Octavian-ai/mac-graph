
import tensorflow as tf

from ..util import *
from ..attention import *

def control_cell(args, features, inputs, in_control_state, in_question_state, in_question_tokens):
	"""
	Build a control cell

	Data in time-minor format

	Arguments:
		- in_control_state.shape   = [batch_size, control_width]
		- in_question_state.shape  = [batch_size, embed_width]
		- in_question_tokens.shape = [batch_size, seq_len, embed_width]

	"""
	with tf.name_scope("control_cell"):

		control_shape = [ features["d_batch_size"], args["control_width"] ]
		in_control_state = dynamic_assert_shape(in_control_state, control_shape)
		
		all_input = tf.concat([in_control_state, inputs], -1, name="all_input")

		question_token_query = tf.layers.dense(all_input, args["embed_width"])
		question_token_query = dynamic_assert_shape(question_token_query, 
			[ features["d_batch_size"], args["embed_width"] ]
		)

		control_out, control_taps = attention(in_question_tokens, question_token_query, 
			word_size=args["embed_width"], output_taps=True)

		if args["control_width"] != args["embed_width"]:
			control_out = tf.layers.dense(control_out, args["control_width"], name="resize_control_out")
		
		control_out = dynamic_assert_shape(control_out, control_shape)

		return control_out, control_taps

