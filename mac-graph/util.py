
import tensorflow as tf

def assert_shape(tensor, shape, batchless=False):

	read_from = 0 if batchless else 1

	assert tensor.shape[read_from:] == shape, f"{tensor.name} is wrong shape, expected {shape} found {tensor.shape[read_from:]}"