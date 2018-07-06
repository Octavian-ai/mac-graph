

import tensorflow as tf

def write_cell(args, in_memory_state, in_data_read, in_control):

	assert_shape(in_memory_state, [args["bus_width"]])
	assert_shape(in_data_read,    [args["bus_width"]])

	all_memory_data = tf.concat([in_memory_state, in_data_read], -1)

	new_memory = tf.layer.dense(all_memory_data, args["bus_width"])

	forget_scalar = tf.layer.dense(in_control, [1])

	out_memory_state = (new_memory * forget_scalar) + (in_memory_state * (1-forget_scalar))
	assert_shape(out_memory_state, [args["bus_width"]])

	return out_memory_state