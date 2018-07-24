
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

		token_full_width = args["embed_width"] + args["pos_enc_width"]

		attention_calls = []

		for i in range(args["control_heads"]):
			question_token_query = tf.layers.dense(all_input, token_full_width)
			question_token_query = tf.layers.dense(question_token_query, token_full_width)
			question_token_query = dynamic_assert_shape(question_token_query, 
				[ features["d_batch_size"], token_full_width ]
			)

			a = attention(in_question_tokens, question_token_query, 
				word_size=token_full_width, 
				output_taps=True,
				max_len=args["max_seq_len"])

			attention_calls.append(a)

		control_out  = [i[0] for i in attention_calls]
		control_out  = tf.concat(control_out, -1)

		control_taps = [i[0] for i in attention_calls]
		control_taps = tf.concat(control_taps, -1)

		if control_out.shape[-1] != args["control_width"]:
			control_out = tf.layers.dense(control_out, args["control_width"], name="resize_control_out")
		
		control_out = dynamic_assert_shape(control_out, control_shape)

		return control_out, control_taps

