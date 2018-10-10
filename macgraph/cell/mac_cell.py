
import tensorflow as tf

from .read_cell import *
from .memory_cell import *
from .control_cell import *
from .output_cell import *
from .write_cell import *
from .messaging_cell import *

from ..util import *
from ..minception import *




class MACCell(tf.nn.rnn_cell.RNNCell):

	def __init__(self, args, features, question_state, question_tokens, vocab_embedding):
		self.args = args
		self.features = features
		self.question_state = question_state
		self.question_tokens = question_tokens
		self.vocab_embedding = vocab_embedding

		super().__init__(self)

	def get_taps(self):
		return {
			"finished":					1,
			"question_word_attn": 		self.args["control_heads"] * self.features["d_src_len"],
			"question_word_attn_raw": 	self.args["control_heads"] * self.features["d_src_len"],
			"kb_node_attn": 			self.args["kb_node_width"] * self.args["embed_width"],
			"kb_node_word_attn": 		self.args["kb_node_width"],
			"kb_edge_attn": 			self.args["kb_edge_width"] * self.args["embed_width"], 
			"kb_edge_word_attn": 		self.args["kb_edge_width"], 
			"read_head_attn": 			2 * self.args["read_heads"],
			"read_head_attn_focus": 	2 * self.args["read_heads"],
			"mp_read_attn": 			self.args["kb_node_max_len"],
			"mp_write_attn": 			self.args["kb_node_max_len"],
			"mp_node_state":			tf.TensorShape([self.args["kb_node_max_len"], self.args["mp_state_width"]]),
			"mp_write_query":			self.args["kb_node_width"] * self.args["embed_width"],	
			"mp_write_signal":			self.args["mp_state_width"],
			"mp_read0_signal":			self.args["mp_state_width"],

		}



	def build_cell(self, inputs, in_state):
		
		with tf.variable_scope("mac_cell", reuse=tf.AUTO_REUSE):

			in_control_state, in_memory_state, in_data_stack, in_mp_state = in_state

			empty_attn = tf.fill([self.features["d_batch_size"], self.features["d_src_len"], 1], 0.0)
			empty_query = tf.fill([self.features["d_batch_size"], self.features["d_src_len"]], 0.0)

			if self.args["use_control_cell"]:
				out_control_state, control_taps = control_cell(self.args, self.features, 
					inputs, in_control_state, self.question_state, self.question_tokens)
			else:
				out_control_state = in_control_state
		
			if self.args["use_read_cell"]:
				read, read_taps = read_cell(
					self.args, self.features, self.vocab_embedding,
					in_memory_state, out_control_state, in_data_stack, 
					self.question_tokens, self.question_state)
			else:
				read = tf.fill([self.features["d_batch_size"], 1], 0.0)
				read_taps = {}


			if self.args["use_message_passing"]:
				mp_reads, out_mp_state, mp_taps = messaging_cell(self.args, self.features, self.vocab_embedding,
					in_mp_state, out_control_state, self.question_state)
			else:
				out_mp_state = in_mp_state
				mp_reads = [tf.fill([self.features["d_batch_size"], self.args["mp_state_width"]], 0.0)]
				mp_taps = {}
			
			if self.args["use_memory_cell"]:
				out_memory_state, tap_memory_forget = memory_cell(self.args, self.features,
					in_memory_state, read, mp_reads, out_control_state)
			else:
				out_memory_state = in_memory_state
				tap_memory_forget = tf.fill([self.features["d_batch_size"], 1], 0.0)

			if self.args["use_data_stack"]:
				out_data_stack = write_cell(self.args, self.features, 
					out_memory_state, read, out_control_state, in_data_stack)
			else:
				out_data_stack = in_data_stack

			
			if self.args["use_output_cell"]:
				output, finished = output_cell(self.args, self.features,
					self.question_state, out_memory_state, read, out_control_state, mp_reads)	
			else:
				output = tf.concat([read, mp_read], -1)
				finished = tf.fill([features["d_batch_size"]], False)

			out_state = (
				out_control_state, 
				out_memory_state, 
				out_data_stack, 
				out_mp_state,
			)

			out_taps = {
				"finished":					tf.cast(finished, tf.float32),
				"question_word_attn": 		control_taps.get("attn", empty_attn),
				"question_word_attn_raw": 	control_taps.get("attn_raw", empty_attn),
				"kb_node_attn": 			tf.squeeze(read_taps.get("kb_node_attn", empty_attn), 2),
				"kb_node_word_attn": 		read_taps.get("kb_node_word_attn", empty_query),
				"kb_edge_attn": 			tf.squeeze(read_taps.get("kb_edge_attn", empty_attn), 2),
				"kb_edge_word_attn": 		read_taps.get("kb_edge_word_attn", empty_query),
				"read_head_attn": 			read_taps.get("read_head_attn", empty_query),
				"read_head_attn_focus": 	read_taps.get("read_head_attn_focus", empty_query),
				"mp_read_attn": 			mp_taps.get("mp_read0_attn", empty_query),
				"mp_write_attn": 			mp_taps.get("mp_write_attn", empty_query),
				"mp_node_state":			out_mp_state,
				"mp_write_query":			mp_taps.get("mp_write_query", empty_query),
				"mp_write_signal":			mp_taps.get("mp_write_signal", empty_query),
				"mp_read0_signal":			mp_taps.get("mp_read0_signal", empty_query),

			}

			return output, out_taps, out_state



	def __call__(self, inputs, in_state):
		'''Build this cell (part of implementing RNNCell)

		This is a wrapper that marshalls our named taps, to 
		make sure they end up where we expect and are present.
		
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
		'''

		output, out_taps, out_state = self.build_cell(inputs, in_state)

		out_taps_keys = set(out_taps.keys())
		expected_keys = set(self.get_taps().keys())

		assert out_taps_keys <= expected_keys, f"Cell builder must return taps in get_taps(), missing {expected_keys - out_taps_keys}"

		out_data = [output]
		for k in self.get_taps().keys():
			out_data.append(out_taps[k])

		return out_data, out_state



	@property
	def state_size(self):
		"""
		Returns a size tuple
		"""
		return (
			self.args["control_width"], 
			self.args["memory_width"], 
			tf.TensorShape([self.args["data_stack_len"], self.args["data_stack_width"]]),
			tf.TensorShape([self.args["kb_node_max_len"], self.args["mp_state_width"]]),
		)

	@property
	def output_size(self):
		return [
			self.args["output_classes"], 
		] + list(self.get_taps().values())





