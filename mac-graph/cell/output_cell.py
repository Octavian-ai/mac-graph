
import tensorflow as tf

from ..minception import *


def output_cell(args, features, in_question_state, in_memory_state, in_read):

	with tf.name_scope("output_cell"):

		in_all = []
		
		if args["use_question_state"]:
			in_all.append(in_question_state)

		if args["use_memory_cell"]:
			in_all.append(in_memory_state)
		else:
			in_all.append(in_read)

		v = tf.concat(in_all, -1)

		# v = tf.layers.dense(v, args["answer_classes"], activation=args["output_activation"])
		# v = tf.layers.dense(v, args["answer_classes"])

		v = tf.layers.dense(v, args["answer_classes"])
		v = mi_activation(v)

		# Don't do softmax here because the loss fn will apply it

		return v