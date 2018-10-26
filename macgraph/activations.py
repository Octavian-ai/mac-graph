
import tensorflow as tf

from .minception import mi_activation

def absu(x):
	return tf.nn.relu(x) + tf.nn.relu(-x)

# Expand activation args to callables
ACTIVATION_FNS = {
	"tanh": tf.tanh,
	"relu": tf.nn.relu,
	"sigmoid": tf.nn.sigmoid,
	"mi": mi_activation,
	"abs": absu,
	"tanh_abs": lambda x: tf.concat([tf.tanh(x), absu(x)], axis=-1),
	"linear": tf.identity,
	"id": tf.identity,
	"selu": tf.nn.selu,
}