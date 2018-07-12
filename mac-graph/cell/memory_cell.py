
import tensorflow as tf

from ..util import *

def memory_cell(args, in_memory_state, in_data_read, in_control):

	with tf.name_scope("memory_cell"):
		assert_shape(in_memory_state, [args["bus_width"]])
		assert_shape(in_data_read,    [args["bus_width"]])

		in_all = tf.concat([in_memory_state, in_data_read], -1)
		new_memory_state = tf.layers.dense(in_all, args["bus_width"])

		forget_scalar = tf.layers.dense(in_control, 1, activation=tf.nn.tanh)

		out_memory_state = (new_memory_state * forget_scalar) + (in_memory_state * (1-forget_scalar))
		
		assert_shape(out_memory_state, [args["bus_width"]])
		return out_memory_state