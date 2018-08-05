
import tensorflow as tf

from ..util import *

def memory_cell(args, features, in_memory_state, in_data_read, in_control_state):

	with tf.name_scope("memory_cell"):

		memory_shape = [features["d_batch_size"], args["memory_width"]]
		in_memory_state = dynamic_assert_shape(in_memory_state, memory_shape)
		
		in_all = tf.concat([
			in_memory_state, 
			in_data_read
		], -1)

		new_memory_state = deeep(in_all, args["memory_width"], args["memory_transform_layers"])

		# We can run this network without a control cell
		if in_control_state is not None:
			forget_scalar = tf.layers.dense(in_control_state, 1, activation=tf.nn.sigmoid)
		else:
			forget_scalar = tf.layers.dense(in_all, 1, activation=tf.nn.sigmoid)
	
		out_memory_state = (new_memory_state * forget_scalar) + (in_memory_state * (1-forget_scalar))
		out_memory_state = dynamic_assert_shape(out_memory_state, memory_shape)
		return out_memory_state