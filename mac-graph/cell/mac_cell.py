
import tensorflow as tf

from .read_cell import *
from .write_cell import *
from .control_cell import *
from ..util import *

def output_unit(args, features, in_question_state, in_memory_state):

	v = tf.concat([in_question_state, in_memory_state], -1)

	# v = tf.layers.dense(v, args["bus_width"], activation=tf.nn.relu)
	v = tf.layers.dense(v, args["answer_classes"], activation=tf.nn.tanh)
	v = tf.layers.dense(v, args["answer_classes"], activation=tf.nn.tanh)

	# Don't do softmax here because the loss fn will apply it

	return v



class MACCell(tf.nn.rnn_cell.RNNCell):

	def __init__(self, args, features, question_state, question_tokens, vocab_embedding):
		self.args = args
		self.features = features
		self.question_state = question_state
		self.question_tokens = question_tokens
		self.vocab_embedding = vocab_embedding

		super().__init__(self)



	def __call__(self, inputs, state):
		"""Run this RNN cell on inputs, starting from the given state.
		
		Args:
			inputs: **Unused!** `2-D` tensor with shape `[batch_size, input_size]`.
			state: if `self.state_size` is an integer, this should be a `2-D Tensor`
				with shape `[batch_size, self.state_size]`.	Otherwise, if
				`self.state_size` is a tuple of integers, this should be a tuple
				with shapes `[batch_size, s] for s in self.state_size`.
			scope: VariableScope for the created subgraph; defaults to class name.
		Returns:
			A pair containing:
			- Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
			- New state: Either a single `2-D` tensor, or a tuple of tensors matching
				the arity and shapes of `state`.
		"""

		in_control_state, in_memory_state = state

		out_memory_state = in_memory_state

		out_control_state = control_cell(self.args, self.features, 
			in_control_state, self.question_state, self.question_tokens)

		data_read = read_cell(self.args, self.features, 
			in_memory_state, out_control_state, self.vocab_embedding)
		
		# out_memory_state = write_cell(self.args, 
		# 	in_memory_state, data_read, out_control_state)
		
		# output = output_unit(self.args, self.features,
		# 	self.question_state, out_memory_state)

		output = tf.layers.dense(data_read, self.args["answer_classes"])
		

		return output, (out_control_state, out_memory_state)



	@property
	def state_size(self):
		"""
		Returns a size tuple (control_state, memory_state)
		"""
		return (self.args["bus_width"], self.args["bus_width"])

	@property
	def output_size(self):
		return self.args["answer_classes"]




