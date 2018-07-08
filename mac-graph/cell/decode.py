
import tensorflow as tf

from .mac_cell import *
from ..util import assert_shape

def execute_reasoning(args, features, labels, question_tokens, question_state, vocab_embedding):
	with tf.variable_scope("decoder",reuse=tf.AUTO_REUSE) as decoder_scope:

		d_cell = MACCell(args, features, question_state, question_tokens, vocab_embedding)
		
		d_cell_initial = d_cell.zero_state(dtype=tf.float32, batch_size=features["d_batch_size"])

		# label_seq: [batch_size, seq_len]
		label_seq = tf.tile(tf.expand_dims(labels, 1), [0, args["max_decode_iterations"]])
		assert_shape(label_seq, [args["max_decode_iterations"]])

		# label_seq_len: [batch_size]
		label_seq_len = tf.tile([args["max_decode_iterations"]], tf.shape(labels))

		decoder_helper = tf.contrib.seq2seq.TrainingHelper(
			label_seq, 
			sequence_length=label_seq_len
		)

		guided_decoder = tf.contrib.seq2seq.BasicDecoder(
			d_cell,
			decoder_helper,
			d_cell_initial)

		# 'outputs' is a tensor of shape [batch_size, max_time, cell.output_size]
		decoded_outputs, decoded_state, decoded_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
			guided_decoder,
			swap_memory=True,
			maximum_iterations=args["max_decode_iterations"],
			scope=decoder_scope)

		
		# Take the final reasoning step output
		final_output = decoded_outputs.rnn_output[:,-1,:]
		assert_shape(final_output, [args["answer_classes"]])

		return final_output
