
import tensorflow as tf

from .read_cell import *
from .write_cell import *
from .control_cell import *

def mac_cell(args, in_states, in_question_state, in_question_tokens, in_knowledge_base):

	in_control_state, in_memory_state = in_states

	out_control_state = control_cell(args, in_control_state, in_question_state, in_question_tokens)
	data_read = read_cell(args, in_memory_state, out_control_state, in_knowledge_base)
	out_memory_state = write_cell(args, in_memory_state, data_read, out_control_state)

	return (out_control_state, out_memory_state)


def output_unit(args, in_question_state, in_memory_state):

	in_states = tf.concat([in_question_state, in_memory_state], -1)

	v = tf.layer.dense(in_states, args["bus_width"], activation=tf.nn.relu)
	v = tf.layer.dense(v, args["answer_classes"])
	output = tf.nn.softmax(v)

	return output



class MACCell(tf.nn.rnn_cell.RNNCell):

	def __init__(self, args, question_state, question_tokens, knowledge_base):
		self.args = args
		self.question_state = question_state
		self.question_tokens = question_tokens
		self.knowledge_base = knowledge_base

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

		next_control, next_memory = mac_cell(self.args, state, self.question_state, self.question_tokens, self.knowledge_base)
		output = output_unit(self.args, self.question_state, next_memory)

		return output, (next_control, next_memory)



	@property
	def state_size(self):
		"""
		Returns a size tuple (control_state, memory_state)
		"""
		return (self.args["bus_width"], self.args["bus_width"])

	@property
	def output_size(self):
		return self.args["answer_classes"]




