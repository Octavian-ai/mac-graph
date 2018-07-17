
import tensorflow as tf


def output_cell(args, features, in_question_state, in_memory_state, in_data_stack):

	with tf.name_scope("output_cell"):
		in_all = [in_question_state, in_memory_state]
		v = tf.concat(in_all, -1)

		v = tf.layers.dense(v, args["answer_classes"], activation=tf.nn.tanh)
		v = tf.layers.dense(v, args["answer_classes"])
		# Don't do softmax here because the loss fn will apply it

		return v