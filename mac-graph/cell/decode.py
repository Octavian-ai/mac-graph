
import tensorflow as tf

from .mac_cell import *
from ..util import *




def dynamic_decode(args, features, inputs, question_state, question_tokens, labels, vocab_embedding):
	with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE) as decoder_scope:

		d_cell = MACCell(args, features, question_state, question_tokens, vocab_embedding)
		d_cell_initial = d_cell.zero_state(dtype=tf.float32, batch_size=features["d_batch_size"])

		# --------------------------------------------------------------------------
		# Decoding handlers
		# --------------------------------------------------------------------------

		finished_shape = tf.convert_to_tensor([features["d_batch_size"], 1])

		initialize_fn = lambda: (
			tf.fill(value=False, dims=finished_shape), # finished
			inputs[0],  # inputs
		)

		sample_fn = lambda time, outputs, state: tf.constant(0) # sampled output

		def next_inputs_fn(time, outputs, state, sample_ids):
			finished = tf.greater(tf.layers.dense(outputs[0], 1, kernel_initializer=tf.zeros_initializer()), 0.5)
			next_inputs = tf.gather(inputs, time+1)
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


		# Peek into the workings
		taps =  decoded_outputs.rnn_output[1]
		taps = tf.expand_dims(taps, -1)
		# TODO reshape into [batch, decode_step, question_word, 1]
		
		# Take the final reasoning step output
		final_output = decoded_outputs.rnn_output[0][:,-1,:]
		
		return final_output, taps





def static_decode(args, features, inputs, question_state, question_tokens, labels, vocab_embedding):
	with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE) as decoder_scope:

		d_cell = MACCell(args, features, question_state, question_tokens, vocab_embedding)
		d_cell_initial = d_cell.zero_state(dtype=tf.float32, batch_size=features["d_batch_size"])

		# Hard-coded unroll of the reasoning network for simplicity
		states = [(inputs[0], d_cell_initial)]
		for i in range(args["max_decode_iterations"]):
			states.append(d_cell(inputs[i+1], states[-1][1]))

		# print(states)
		final_output = states[-1][0][0]

		taps = [i[0][1] for i in states if i[0] is not None]
		taps = tf.concat(taps, axis=-1)
		taps = tf.transpose(taps, [0,2,1])
		taps = tf.expand_dims(taps, axis=-1)
		
		return final_output, taps


def execute_reasoning(args, features, question_state, question_tokens, **kwargs):

	inputs = [
		tf.layers.dense(question_state, args["control_width"], name=f"question_state_inputs_t{i}") 
		for i in range(args["max_decode_iterations"]+1)
	]

	# [batch, seq, width]
	print(question_tokens)
	# question_tokens_pos = add_location_encoding_1d(question_tokens, dim=args["question_token_pos"])
	# print(question_tokens_pos)
	question_tokens_pos = question_tokens

	if args["use_dynamic_decode"]:
		final_output, taps = dynamic_decode(args, features, inputs, question_state, question_tokens_pos, **kwargs)
	else:
		final_output, taps = static_decode(args, features, inputs, question_state, question_tokens_pos, **kwargs)

	tf.summary.image("Question_words", taps, family="Attention")

	final_output = dynamic_assert_shape(final_output, [features["d_batch_size"], args["answer_classes"]])
	return final_output




