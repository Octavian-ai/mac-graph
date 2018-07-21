
import tensorflow as tf

from .mac_cell import *
from ..util import *




def dynamic_decode(args, features, labels, question_tokens, question_state, vocab_embedding):
	with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE) as decoder_scope:

		d_cell = MACCell(args, features, question_state, question_tokens, vocab_embedding)
		d_cell_initial = d_cell.zero_state(dtype=tf.float32, batch_size=features["d_batch_size"])

		# --------------------------------------------------------------------------
		# Decoding handlers
		# --------------------------------------------------------------------------

		finished_shape = tf.convert_to_tensor([features["d_batch_size"], 1])

		initialize_fn = lambda: (
			tf.fill(value=False, dims=finished_shape), # finished
			tf.constant(0),  # inputs
		)

		sample_fn = lambda time, outputs, state: tf.constant(0) # sampled output

		def next_inputs_fn(time, outputs, state, sample_ids):
			finished = tf.greater(tf.layers.dense(outputs, 1), 0.5)
			next_inputs = tf.constant(0)
			next_state = state
			return (finished, next_inputs, next_state)

		decoder_helper = tf.contrib.seq2seq.CustomHelper(
			initialize_fn, sample_fn, next_inputs_fn
		)

		decoder = tf.contrib.seq2seq.BasicDecoder(
			d_cell,
			decoder_helper,
			d_cell_initial)


		# --------------------------------------------------------------------------
		# Do the decode!
		# --------------------------------------------------------------------------

		# 'outputs' is a tensor of shape [batch_size, max_time, cell.output_size]
		decoded_outputs, decoded_state, decoded_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
			decoder,
			swap_memory=True,
			maximum_iterations=args["max_decode_iterations"],
			scope=decoder_scope)

		
		# Take the final reasoning step output
		final_output = decoded_outputs.rnn_output[:,-1,:]
		assert_shape(final_output, [args["answer_classes"]])
		return final_output





def static_decode(args, features, labels, question_tokens, question_state, vocab_embedding):
	with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE) as decoder_scope:

		d_cell = MACCell(args, features, question_state, question_tokens, vocab_embedding)
		d_cell_initial = d_cell.zero_state(dtype=tf.float32, batch_size=features["d_batch_size"])

		# Hard-coded unroll of the reasoning network for simplicity
		states = [(None, d_cell_initial)]
		for i in range(args["max_decode_iterations"]):
			states.append(d_cell(None, states[-1][1]))

		final_output = states[-1][0]
		final_output = dynamic_assert_shape(final_output, [features["d_batch_size"], args["answer_classes"]])

		return final_output


def execute_reasoning(args, **kwargs):
	if not args["use_dynamic_decode"]:
		return dynamic_decode(args, **kwargs)
	else:
		return static_decode(args, **kwargs)


