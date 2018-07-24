
import tensorflow as tf
import math

def assert_shape(tensor, shape, batchless=False):

	read_from = 0 if batchless else 1

	lhs = tf.TensorShape(tensor.shape[read_from:])
	rhs = tf.TensorShape(shape)

	lhs.assert_is_compatible_with(rhs)
	
	# assert lhs == shape, f"{tensor.name} is wrong shape, expected {shape} found {lhs}"

def assert_rank(tensor, rank):
	assert len(tensor.shape) == rank, f"{tensor.name} is wrong rank, expected {rank} got {len(tensor.shape)}"


def dynamic_assert_shape(tensor, shape):
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

	lhs = tf.shape(tensor)
	rhs = tf.convert_to_tensor(shape, dtype=lhs.dtype)

	assert_op = tf.assert_equal(lhs, rhs, message=f"Asserting shape of {tensor.name}", summarize=10)

	with tf.control_dependencies([assert_op]):
		return tf.identity(tensor, name="dynamic_assert_shape")


def minimize_clipped(optimizer, value, max_gradient_norm):
	global_step = tf.train.get_global_step()
	var = tf.trainable_variables()
	gradients = tf.gradients(value, var)
	clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
	return optimizer.apply_gradients(zip(clipped_gradients, var), global_step=global_step)


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


def vector_to_barcode(tensor):
	width = tf.shape(tensor)[-1]
	barcode_height = tf.cast(tf.round(tf.div(tf.cast(width, tf.float32), 3.0)), tf.int32)
	barcode_image = tf.tile(tf.reshape(tensor, [-1, 1, width, 1]), [1, barcode_height, 1, 1])
	return barcode_image




def add_location_encoding_1d(tensor, dim=32, width_axis=1, concat_axis=2): 
	'''
	The function is based on https://github.com/stanfordnlp/mac-network

	Computes sin/cos positional encoding for h x w x (4*dim). 
	If outDim positive, casts positions to that dimension.
	Based on positional encoding presented in "Attention is all you need"

	Currently hard-coded for one setup of width_axis and concat_axis
	'''   
	locationBias = 1.5

	batch = tf.shape(tensor)[0]
	w = tf.shape(tensor)[width_axis]
	halfdim = dim / 2

	x = tf.expand_dims(tf.to_float(tf.linspace(-locationBias, locationBias, w)), axis=1)
	i = tf.expand_dims(tf.to_float(tf.range(halfdim)), axis=0)

	peSinX = tf.sin(x / (tf.pow(10000.0, i / halfdim)))
	peCosX = tf.cos(x / (tf.pow(10000.0, i / halfdim)))

	pe = tf.concat([peSinX, peCosX], axis=-1)
	pe = tf.expand_dims(pe, 0)
	pe = tf.tile(pe, [batch, 1, 1])
	
	tensor = tf.concat([tensor,pe], axis=concat_axis)
	

	return tensor



