
import tensorflow as tf

from ..util import *
from ..args import ACTIVATION_FNS

def memory_cell(args, features, in_memory_state, in_data_read, in_mp_reads, in_control_state, in_iter_id):

	with tf.name_scope("memory_cell"):

		memory_shape = [features["d_batch_size"], args["memory_width"]]
		in_memory_state = dynamic_assert_shape(in_memory_state, memory_shape)
		
		in_all = [in_memory_state, in_iter_id]
		if args["use_read_cell"]:
			in_all.append(in_data_read)

		if args["use_message_passing"]:
			in_all.extend(in_mp_reads)

		in_all = tf.concat(in_all, -1)

		forget_act = ACTIVATION_FNS[args["memory_forget_activation"]]
		act = ACTIVATION_FNS[args["memory_activation"]]

		new_memory_state = in_all
		for i in range(args["memory_transform_layers"]):
			new_memory_state = tf.layers.dense(new_memory_state, args["memory_width"], activation=act)

		forget_scalar = tf.layers.dense(in_all, 1, activation=forget_act)
	
		out_memory_state = (new_memory_state * forget_scalar) + (in_memory_state * (1-forget_scalar))
		out_memory_state = dynamic_assert_shape(out_memory_state, memory_shape)
		return out_memory_state, forget_scalar