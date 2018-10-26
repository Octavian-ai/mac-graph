
import tensorflow as tf
import math
from .args import ACTIVATION_FNS

def tf_assert_almost_equal(x, y, delta=0.001, **kwargs):
	return tf.assert_less(tf.abs(x-y), delta, **kwargs)

def assert_shape(tensor, shape, batchless=False):

	read_from = 0 if batchless else 1

	lhs = tf.TensorShape(tensor.shape[read_from:])
	rhs = tf.TensorShape(shape)

	lhs.assert_is_compatible_with(rhs)
	
	# assert lhs == shape, f"{tensor.name} is wrong shape, expected {shape} found {lhs}"

def assert_rank(tensor, rank):
	assert len(tensor.shape) == rank, f"{tensor.name} is wrong rank, expected {rank} got {len(tensor.shape)}"


def dynamic_assert_shape(tensor, shape, name=None):
	"""
	Check that a tensor has a shape given by a list of constants and tensor values.

	This function will place an operation into your graph that gets executed at runtime.
	This is helpful because often tensors have many dynamic sized dimensions that
	you cannot otherwise compare / assert are as you expect.

	For example, measure a dimension at run time:
	`batch_size = tf.shape(my_tensor)[0]`
	
	then assert another tensor does indeed have the right shape:  
	`other_tensor = dynamic_assert_shape(other_tensor, [batch_size, 16])`

	You should use this as an inline identity function so that the operation it generates
	gets added and executed in the graph

	Returns: the argument `tensor` unchanged
	"""

	tensor_shape = tf.shape(tensor)
	tensor_shape = tf.cast(tensor_shape, tf.int64)
	
	expected_shape = tf.convert_to_tensor(shape)
	expected_shape = tf.cast(expected_shape, tf.int64)
	
	t_name = "tensor" if tf.executing_eagerly() else tensor.name

	if isinstance(shape, list):
		assert len(tensor.shape) == len(shape), f"Tensor shape {tensor_shape} and expected shape {expected_shape} have different lengths"

	assert_op = tf.assert_equal(tensor_shape, expected_shape, message=f"Asserting shape of {t_name}", summarize=10, name=name)

	with tf.control_dependencies([assert_op]):
		return tf.identity(tensor, name="dynamic_assert_shape")



def minimize_clipped(optimizer, value, max_gradient_norm, var=None):
	global_step = tf.train.get_global_step()

	if var is None:
		var = tf.trainable_variables()
	
	gradients = tf.gradients(value, var)
	clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
	grad_dict = dict(zip(var, clipped_gradients))
	op = optimizer.apply_gradients(zip(clipped_gradients, var), global_step=global_step)
	return op, grad_dict


def vector_to_barcode(tensor):
	width = tf.shape(tensor)[-1]
	barcode_height = tf.cast(tf.round(tf.div(tf.cast(width, tf.float32), 3.0)), tf.int32)
	barcode_image = tf.tile(tf.reshape(tensor, [-1, 1, width, 1]), [1, barcode_height, 1, 1])
	return barcode_image




def add_positional_encoding_1d(tensor, seq_axis=1, word_axis=2, dtype=tf.float32): 
	'''
	The function is based on https://github.com/stanfordnlp/mac-network

	Computes sin/cos positional encoding for h x w x (4*dim). 
	If outDim positive, casts positions to that dimension.
	Based on positional encoding presented in "Attention is all you need"

	Currently hard-coded for one setup of seq_axis and word_axis
	'''   

	assert len(tensor.shape) == 3, "Expecting tensor of shape [batch, seq, word]"

	in_tensor_shape = tf.shape(tensor)

	batch_len = tf.shape(tensor)[0]
	seq_len = tf.shape(tensor)[seq_axis]
	word_len = tf.shape(tensor)[word_axis]
	
	halfdim = tf.cast(word_len / 2, dtype)

	x = tf.expand_dims(tf.to_float(tf.range(seq_len)), axis=1)
	i = tf.expand_dims(tf.to_float(tf.range(halfdim)), axis=0)

	peSinX = tf.sin(x / (tf.pow(10000.0, i / halfdim)))
	peCosX = tf.cos(x / (tf.pow(10000.0, i / halfdim)))

	pe = tf.concat([peSinX, peCosX], axis=-1)
	pe = tf.expand_dims(pe, 0)
	# pe = tf.tile(pe, [batch, 1, 1])
	# pe = dynamic_assert_shape(pe, tf.shape(tensor))

	# Original paper
	tensor = tensor + pe
	tensor = dynamic_assert_shape(tensor, in_tensor_shape)
	
	# Concat method
	# tensor = tf.concat([tensor,pe], axis=word_axis)
	

	return tensor



