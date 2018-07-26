
import tensorflow as tf

from .read_cell import *
from .memory_cell import *
from .control_cell import *
from .output_cell import *
from .write_cell import *
from ..util import *



class MACCell(tf.nn.rnn_cell.RNNCell):

	def __init__(self, args, features, question_state, question_tokens, vocab_embedding):
		self.args = args
		self.features = features
		self.question_state = question_state
		self.question_tokens = question_tokens
		self.vocab_embedding = vocab_embedding

		super().__init__(self)



	def __call__(self, inputs, in_state):
		"""Run this RNN cell on inputs, starting from the given state.
		
		Args:
			inputs: `2-D` tensor with shape `[batch_size, input_size]`.
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


		with tf.variable_scope("mac_cell", reuse=tf.AUTO_REUSE):

			in_control_state, in_memory_state, in_data_stack = in_state

			if self.args["use_control_cell"]:
				out_control_state, tap_question_attn, tap_question_query = control_cell(self.args, self.features, 
					inputs, in_control_state, self.question_state, self.question_tokens)
			else:
				out_control_state = in_control_state
				tap_question_attn = None
				tap_question_query = None

		
			read, tap_read_attn, tap_read_table = read_cell(self.args, self.features, self.vocab_embedding,
				in_memory_state, out_control_state, in_data_stack, self.question_tokens)
		
			
			if self.args["use_memory_cell"]:
				out_memory_state = memory_cell(self.args, self.features,
					in_memory_state, read, out_control_state)
			else:
				out_memory_state = in_memory_state

			if self.args["use_data_stack"]:
				out_data_stack = write_cell(self.args, self.features, 
					out_memory_state, read, out_control_state, in_data_stack)
			else:
				out_data_stack = in_data_stack
			
		
			output = output_cell(self.args, self.features,
				self.question_state, out_memory_state, read)	

			out_state = (out_control_state, out_memory_state, out_data_stack)
			out_data  = (output, 
				tap_question_attn, tap_question_query,
				tap_read_attn, tap_read_table,
				out_control_state)

			return out_data, out_state



	@property
	def state_size(self):
		"""
		Returns a size tuple (control_state, memory_state)
		"""
		return (
			self.args["control_width"], 
			self.args["memory_width"], 
			tf.TensorShape([self.args["data_stack_len"], self.args["data_stack_width"]]),
		)

	@property
	def output_size(self):

		read_attn_width = 0
		for i in ["kb_edge", "kb_node"]:
			if self.args["use_"+i]:
				read_attn_width += self.args[i+"_width"] * self.args["embed_width"]
		

		return (
			self.args["answer_classes"],
			self.features["d_seq_len"],
			self.features["d_seq_len"],
			read_attn_width,
			self.args["control_width"],
		)





