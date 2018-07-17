
import tensorflow as tf

from ..util import *


def write_cell(args, features, in_memory_state, in_data_read, in_control_state, in_data_stack):
	"""
	Returns: The updated data stack
	"""
	with tf.name_scope("write_cell"):

		stack_shape = [features["d_batch_size"], args["data_stack_len"], args["data_stack_width"]]
		in_data_stack = dynamic_assert_shape(in_data_stack, stack_shape)
				
		def do_pop(in_data_stack):

			blank = tf.zeros([features["d_batch_size"], 1, args["data_stack_width"]])
			headless = in_data_stack[:, 1:, :]

			return tf.concat([headless, blank], axis=1)

		def do_push(in_data_stack, in_value):
			in_value = dynamic_assert_shape(in_value, [features["d_batch_size"], args["data_stack_width"]])
			in_value = tf.expand_dims(in_value, 1)
			shortened = in_data_stack[:, :-1, :]
			return tf.concat([in_value, shortened], axis=1)


		in_all = [in_memory_state, in_control_state]
		in_all = tf.concat(in_all, axis=-1)

		# Pop bool
		pop_signal = tf.layers.dense(in_all, 1, activation=tf.nn.sigmoid)
		
		# Push bool
		push_signal = tf.layers.dense(in_all, 1, activation=tf.nn.sigmoid)
		push_value = deeep(in_all, args["data_stack_width"])

		
		# Execute!
		s = in_data_stack
		s = pop_signal * do_pop(s) + (1-pop_signal) * s
		s = dynamic_assert_shape(s, stack_shape)

		s = push_signal * do_push(s, push_value) + (1-push_signal) * s
		s = dynamic_assert_shape(s, stack_shape)

		return s

