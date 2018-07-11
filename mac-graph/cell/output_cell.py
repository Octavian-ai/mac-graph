
import tensorflow as tf


def output_cell(args, features, in_question_state, in_memory_state):

	v = tf.concat([in_question_state, in_memory_state], -1)

	# v = tf.layers.dense(v, args["bus_width"], activation=tf.nn.relu)
	v = tf.layers.dense(v, args["answer_classes"], activation=tf.nn.tanh)
	v = tf.layers.dense(v, args["answer_classes"])

	# Don't do softmax here because the loss fn will apply it

	return v