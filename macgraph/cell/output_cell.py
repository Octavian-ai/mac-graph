
import tensorflow as tf

from ..minception import *


def output_cell(args, features, in_question_state, in_memory_state, in_read, in_control_state):

	with tf.name_scope("output_cell"):

		in_all = []
		
		if args["use_question_state"]:
			in_all.append(in_question_state)

		if args["use_memory_cell"]:
			in_all.append(in_memory_state)
		
		if len(in_all) == 0:
			in_all.append(in_read)

		v = tf.concat(in_all, -1)

		# v = tf.layers.dense(v, args["answer_classes"], activation=args["output_activation"])
		# v = tf.layers.dense(v, args["answer_classes"])

		mi_control = in_control_state if args["use_control_cell"] else None

		for i in range(args["output_layers"]):
			v = tf.layers.dense(v, args["answer_classes"])
			v = mi_activation(v, control=mi_control)

		# Don't do softmax here because the loss fn will apply it

		return v