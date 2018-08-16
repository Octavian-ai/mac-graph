

import tensorflow as tf
import math

from ..util import *

def basic_cell(args, i, width):

	c = tf.contrib.rnn.LSTMCell(width)
	c = tf.contrib.rnn.DropoutWrapper(c, args['input_dropout'])

	if args["use_input_residual"]:
		assert i > 0 or args["embed_width"] == args["input_width"], "Cannot use residual on base layer as inputs and outputs are of different width"
		c = tf.contrib.rnn.ResidualWrapper(c)

	return c

def cell_stack(args, width, layer_mul=1):
	cells = []
	for i in range(int(args["input_layers"]*layer_mul)):
		cells.append(basic_cell(args, i, width))

	cell = tf.contrib.rnn.MultiRNNCell(cells)
	return cell


def encode_input(args, features, vocab_embedding):
	"""
	Expects data in time-minor format (e.g. batch-major)

	Arguments:
		- features {
			"src":     Tensor([batch_size, seq_len]), 
			"src_len": Tensor([batch_size])
		}

	Returns:
		(
			Tensor("outputs",     [batch_size, seq_len, control_width * 2]), 
			Tensor("final_state", [batch_size, control_width * 2])
		)
	"""
	with tf.name_scope("encoder"):
		
		# --------------------------------------------------------------------------
		# Setup inputs
		# --------------------------------------------------------------------------

		assert "src" in features
		assert "src_len" in features

		batch_size = features["d_batch_size"]
		seq_len    = features["d_src_len"]

		# Trim down to the residual batch size (e.g. when at end of input data)
		padded_src_len = features["src_len"][0 : batch_size]

		# --------------------------------------------------------------------------
		# Embed vocab
		# --------------------------------------------------------------------------

		src  = tf.nn.embedding_lookup(vocab_embedding, features["src"])
		src = dynamic_assert_shape(src, [batch_size, seq_len, args["embed_width"]])

		# --------------------------------------------------------------------------
		# Encoder
		# --------------------------------------------------------------------------
		
		# 1/2 multiplier so that when we concat the layers together we get control_width
		fw_cell = cell_stack(args, width=math.floor(args['input_width']/2))
		bw_cell = cell_stack(args, width=math.ceil(args['input_width']/2))
		
		(fw_output, bw_output), (fw_states, bw_states) = tf.nn.bidirectional_dynamic_rnn(
			fw_cell,
			bw_cell,
			src,
			dtype=tf.float32,
			# sequence_length=padded_src_len, # was causing seg fault 11
			swap_memory=True)

		
		question_tokens = tf.concat( (fw_output, bw_output), axis=-1)
		question_tokens = dynamic_assert_shape(question_tokens, 
			[ features["d_batch_size"], features["d_src_len"], args["input_width"] ]
		)

		# Top layer, output layer
		question_state = tf.concat( (fw_states[-1].c, bw_states[-1].c), axis=-1)
		question_state = dynamic_assert_shape(question_state,
			[ features["d_batch_size"], args["input_width"] ]
		)

		return (question_tokens, question_state)





