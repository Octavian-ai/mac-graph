
import tensorflow as tf

from .activations import *

def layer_selu(tensor, width, dropout=0.0, name=None):

	if name is None:
		name_dense = None
		name_drop = None
	else:
		name_dense = name + "_dense"
		name_drop = name + "_drop"

	r = tf.layers.dense(tensor, width, 
		activation=tf.nn.selu,
		kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0), 
		name=name_dense)

	if dropout > 0.0:
		r = tf.contrib.nn.alpha_dropout(r, dropout, name=name_drop)

	return r

def layer_dense(tensor, width, activation_str, dropout=0.0, name=None):

	if activation_str == "selu":
		return layer_selu(tensor, width, dropout, name)
	else:
		v =  tf.layers.dense(tensor, width, activation=ACTIVATION_FNS[activation_str], name=name)

		if dropout > 0:
			v = tf.nn.dropout(v, 1.0-dropout)

		return v


def deeep(tensor, width, depth=2, residual_depth=3, activation=tf.nn.tanh):
	"""
	Quick 'n' dirty "let's slap on some layers" function. 

	Implements residual connections and applys them when it can. Uses this schematic:
	https://blog.waya.ai/deep-residual-learning-9610bb62c355
	"""
	with tf.name_scope("deeep"):

		if residual_depth is not None:
			for i in range(math.floor(depth/residual_depth)):
				tensor_in = tensor

				for j in range(residual_depth-1):
					tensor = tf.layers.dense(tensor, width, activation=activation)

				tensor = tf.layers.dense(tensor, width)
			
				if tensor_in.shape[-1] == width:
					tensor += tensor_in
			
				tensor = activation(tensor)

			remaining = depth % residual_depth
		else:
			remaining = depth

		for i in range(remaining):
			tensor = tf.layers.dense(tensor, width, activation=activation)

		return tensor