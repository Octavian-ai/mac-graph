
import tensorflow as tf

from .mac_cell import *
from ..util import *


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def expand_if_needed(t, target=4):
	if t is None:
		return t

	while len(t.shape) < target:
		t = tf.expand_dims(t, -1)
	return t


# --------------------------------------------------------------------------
# RNN loops
# --------------------------------------------------------------------------

def dynamic_decode(args, features, inputs, question_state, question_tokens, labels, vocab_embedding):
	with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE) as decoder_scope:

		d_cell = MACCell(args, features, question_state, question_tokens, vocab_embedding)
		d_cell_initial = d_cell.zero_state(dtype=tf.float32, batch_size=features["d_batch_size"])
		
		# --------------------------------------------------------------------------
		# Decoding handlers
		# --------------------------------------------------------------------------

		finished_shape = tf.convert_to_tensor([features["d_batch_size"], 1])

		def get_iteration_id(time):
			return tf.tile(tf.expand_dims(tf.one_hot(time, args["max_decode_iterations"]), 0), [features["d_batch_size"], 1])

		def get_input_for_time(time):
			time = dynamic_assert_shape(time, [], "time")
			return (tf.gather(inputs, time), get_iteration_id(time))

		initialize_fn = lambda: (
			tf.fill(value=False, dims=finished_shape), # finished
			get_input_for_time(tf.constant(0)),  # inputs
		)

		sample_fn = lambda time, outputs, state: tf.constant(0) # sampled output

		def next_inputs_fn(time, outputs, state, sample_ids):
			if args["use_early_stopping"]:
				finished = tf.greater(outputs[1], 0.9)
			else:
				finished = tf.constant(False, [features["d_batch_size"]])
			next_inputs = get_input_for_time(time+1)
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

		out_taps = {
			key: decoded_outputs.rnn_output[idx+1]
			for idx, key in enumerate(d_cell.get_taps().keys())
		}

		out_taps["decode_iterations"] = decoded_sequence_lengths
		
		# Take the final reasoning step output
		outputs = decoded_outputs.rnn_output[0]
		final_output = outputs[:,-1,:]

		return outputs, final_output, out_taps





def static_decode(args, features, inputs, question_state, question_tokens, labels, vocab_embedding):
	with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):

		d_cell = MACCell(args, features, question_state, question_tokens, vocab_embedding)
		d_cell_initial = d_cell.zero_state(dtype=tf.float32, batch_size=features["d_batch_size"])

		# Hard-coded unroll of the reasoning network for simplicity
		states = [(None, d_cell_initial)]
		for i in range(args["max_decode_iterations"]):
			with tf.variable_scope("decoder_cell", reuse=tf.AUTO_REUSE):
				states.append(d_cell(inputs[i], states[-1][1]))

		final_output = states[-1][0][0]

		def get_tap(idx, key):
			with tf.name_scope(f"get_tap_{key}"):
				tap = [i[0][idx] for i in states if i[0] is not None]
				for i in tap:
					if i is None:
						return None

				tap = tf.convert_to_tensor(tap)

				 # Deal with batch vs iteration axis layout
				if len(tap.shape) == 3:
					tap = tf.transpose(tap, [1,0,2]) # => batch, iteration, data
				if len(tap.shape) == 4:
					tap = tf.transpose(tap, [1,0,2,3]) # => batch, iteration, control_head, data
					
				return tap

		out_taps = {
			key: get_tap(idx+1, key)
			for idx, key in enumerate(d_cell.get_taps().keys())
		}
		
		return final_output, out_taps


def execute_reasoning(args, features, question_state, question_tokens, **kwargs):

	question_state_per_iteration = [
		tf.layers.dense(question_state, args["control_width"], name=f"question_state_inputs_t{i}") 
		for i in range(args["max_decode_iterations"]+1)
	]

	iteration_id = tf.eye(args["max_decode_iterations"])

	# inputs = tf.stack([question_state_per_iteration, iteration_id], axis=-1)
	# print(inputs)
	inputs = question_state_per_iteration

	if args["use_position_encoding"]:
		question_tokens = add_location_encoding_1d(question_tokens)

	if args["use_dynamic_decode"]:
		return dynamic_decode(args, features, inputs, question_state, question_tokens, **kwargs)
	else:
		return static_decode(args, features, inputs, question_state, question_tokens, **kwargs)


	




