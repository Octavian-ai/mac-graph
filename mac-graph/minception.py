
import tensorflow as tf

from .util import dynamic_assert_shape
from .const import EPSILON

'''
The mini-inception (mi) library

This is inspired by Google's inception network 
and DARTS architecture search. I didn't get fancy
on the bilevel optimization, so let's see how it goes!!

'''


def mi_activation(tensor, tap=False):
	with tf.name_scope("mi_activation"):
		activations = [
			tf.tanh, 
			tf.nn.sigmoid, 
			tf.nn.relu, 
			tf.identity, 
			lambda x: tf.nn.relu(x) + tf.nn.relu(-x)
		]

		choice = tf.get_variable("activation_choice", [len(activations)])
		choice = tf.nn.softmax(choice)

		t = [activations[i](tensor) * choice[i] 
				for i in range(len(activations))]
		t = sum(t)

		t = dynamic_assert_shape(t, tf.shape(tensor))

		if tap:
			return t, choice
		else:
			return t


def mi_residual(tensor, width):
	with tf.name_scope("mi_residual"):

		choice = tf.get_variable("choice", [2])
		choice = tf.nn.sigmoid(choice)

		tensor = tf.layers.dense(tensor, width)

		left = choice[0] * tf.layers.dense(
			mi_activation(
				tf.layers.dense(tensor, width)
			)
		, width)

		right = choice[1] * tensor

		join = left + right
		out = mi_activation(join)

		return join



def mi_deep(tensor, width, depth):
	with tf.name_scope("mi_deep"):

		t = tensor

		for i in range(depth // 2):
			t = mi_residual(t, width)

		for i in range(depth % 2):
			t = tf.layers.dense(t, width)
			t = mi_activation(t)

		return t




