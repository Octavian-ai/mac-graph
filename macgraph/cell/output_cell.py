
import tensorflow as tf

from ..minception import *
from ..args import ACTIVATION_FNS
from ..util import *


def output_cell(args, features, in_question_state, in_memory_state, in_read, in_control_state, in_mp_reads):

	with tf.name_scope("output_cell"):

		in_all = []
		
		if args["use_question_state"]:
			in_all.append(in_question_state)

		if args["use_memory_cell"]:
			in_all.append(in_memory_state)
		
		if args["use_output_read"]:
			in_all.append(in_read)

		if args["use_message_passing"]:
			in_all.extend(in_mp_reads)

		in_all = tf.concat(in_all, -1)

		v = in_all
		finished = in_all

		for i in range(args["output_layers"]):
			# v = tf.layers.dense(v, args["output_classes"])
			# v = ACTIVATION_FNS[args["output_activation"]](v)
			v = layer_selu(v, args["output_classes"])
			finished = layer_selu(v, in_all.shape[-1]/4)

		finished = tf.greater(tf.layers.dense(finished, 1, kernel_initializer=tf.zeros_initializer()), 0.5)

		return v, finished