
import tensorflow as tf

from ..minception import *
from ..args import ACTIVATION_FNS


def output_cell(args, features, in_question_state, in_memory_state, in_read, in_control_state):

	with tf.name_scope("output_cell"):

		in_all = []
		
		if args["use_question_state"]:
			in_all.append(in_question_state)

		if args["use_memory_cell"]:
			in_all.append(in_memory_state)
		else:
			in_all.append(in_read)

		v = tf.concat(in_all, -1)

		for i in range(args["output_layers"]):
			v = tf.layers.dense(v, args["answer_classes"])
			v = ACTIVATION_FNS[args["output_activation"]](v)

		v = tf.layers.dense(v, args["answer_classes"])

		return v