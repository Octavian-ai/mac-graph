
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
		
		all_input = tf.concat([in_question_state, in_control_state, inputs], -1, name="all_input")

		question_token_width = args["input_width"]

		attention_calls = []
		queries = []

		for i in range(args["control_heads"]):
			question_token_query = tf.layers.dense(all_input, question_token_width)
			question_token_query = tf.layers.dense(question_token_query, question_token_width)
			question_token_query = dynamic_assert_shape(question_token_query, 
				[ features["d_batch_size"], question_token_width ]
			)
			queries.append(question_token_query)

			a = attention(
				table=in_question_tokens, 
				query=question_token_query, 
				key_width=question_token_width, 
				keys_len=features["src_len"],
			)

			attention_calls.append(a)

		control_out  = [i[0] for i in attention_calls]
		control_out  = tf.concat(control_out, -1)

		tap_qw_attn = [i[2]["attn"] for i in attention_calls]
		tap_qw_attn = tf.concat(tap_qw_attn, -1) # Concat the unitary score dimensions
		tap_qw_attn = tf.transpose(tap_qw_attn, [0, 2, 1]) # switch so last dimension is words

		if control_out.shape[-1] != args["control_width"]:
			control_out = tf.layers.dense(control_out, args["control_width"], name="resize_control_out")
		
		control_out = tf.nn.dropout(control_out, 1.0-args["control_dropout"])
		control_out = dynamic_assert_shape(control_out, control_shape)

		return control_out, tap_qw_attn

