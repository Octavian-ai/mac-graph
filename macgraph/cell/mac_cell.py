
import tensorflow as tf

from ..component import Component

from .read_cell import *
from .memory_cell import *
from .control_cell import *
from .output_cell import *
from .write_cell import *
from .messaging_cell import *
from .types import *

from ..util import *
from ..minception import *
from ..layers import *

class MAC_RNNCell(tf.nn.rnn_cell.RNNCell):

	def __init__(self, args, features, question_state, question_tokens, vocab_embedding):

		self.args = args
		self.features = features
		self.question_state = question_state
		self.question_tokens = question_tokens
		self.vocab_embedding = vocab_embedding

		self.mac = MAC_Component()

		super().__init__(self)


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

		output, out_state = self.mac.forward(self.args, self.features, 
			inputs, in_state, 
			self.question_state, self.question_tokens, self.vocab_embedding)

		taps = self.mac.all_taps()

		out_data = [output]

		for k,v in taps.items():
			out_data.append(v.tensor)

		return out_data, out_state



	@property
	def state_size(self):
		"""
		Returns a size tuple
		"""
		return (
			self.args["control_width"], 
			self.args["memory_width"], 
			tf.TensorShape([self.args["kb_node_max_len"], self.args["mp_state_width"]]),
		)

	@property
	def output_size(self):

		tap_sizes = self.mac.all_tap_sizes()

		return [
			self.args["output_width"], 
		] + tap_sizes




class MAC_Component(Component):

	def __init__(self, ):
		super().__init__("mac_cell")

	"""
	Special forward. Should return output, out_state
	"""
	def forward(self, args, features, inputs, in_state, question_state, question_tokens, vocab_embedding):
		# TODO: remove this transition scaffolding
		self.args = args
		self.features = features

		self.question_state = question_state
		self.question_tokens = question_tokens
		self.vocab_embedding = vocab_embedding

		with tf.variable_scope("mac_cell", reuse=tf.AUTO_REUSE):

			in_control_state, in_memory_state, in_node_state = in_state

			in_iter_question_state = inputs[0]
			in_iter_question_state = dynamic_assert_shape(in_iter_question_state, [self.features["d_batch_size"], self.args["control_width"]], "in_iter_question_state")
			
			in_iter_id = inputs[1]
			in_iter_id = dynamic_assert_shape(in_iter_id, [self.features["d_batch_size"], self.args["max_decode_iterations"]], "in_iter_id")

			in_prev_outputs = inputs[-1]

			if self.args["use_control_cell"]:
				out_control_state, control_taps = control_cell(self.args, self.features, 
					in_iter_question_state, in_control_state, self.question_state, self.question_tokens)
			else:
				out_control_state = in_control_state
				control_taps = {}

			context = CellContext(
				features=self.features, 
				args=self.args,
				vocab_embedding=self.vocab_embedding,
				in_prev_outputs=in_prev_outputs,
				in_iter_id=in_iter_id,
				in_iter_question_state=in_iter_question_state,
				in_memory_state=in_memory_state,
				in_question_tokens=self.question_tokens,
				in_question_state=self.question_state,
				in_node_state=in_node_state,
				control_state=out_control_state,
			)
		
			if self.args["use_read_cell"]:
				reads = []
				read_taps = {}
				for head_index in range(self.args["read_heads"]):
					read, read_taps_ = read_cell(context, head_index,
						self.args, self.features, self.vocab_embedding,
						in_memory_state, out_control_state, in_prev_outputs,
						self.question_tokens, self.question_state, 
						in_iter_id)

					reads.append(read)
					read_taps = {**read_taps, **read_taps_}
			else:
				reads = [tf.fill([self.features["d_batch_size"], 1], 0.0)]
				read_taps = {}


			if self.args["use_message_passing"]:
				mp_reads, out_mp_state, mp_taps = messaging_cell(context)
			else:
				out_mp_state = in_node_state
				mp_reads = [tf.fill([self.features["d_batch_size"], self.args["mp_state_width"]], 0.0)]
				mp_taps = {}
			
			if self.args["use_memory_cell"]:
				out_memory_state, tap_memory_forget = memory_cell(self.args, self.features,
					in_memory_state, reads, mp_reads, out_control_state, in_iter_id)
			else:
				out_memory_state = in_memory_state
				tap_memory_forget = tf.fill([self.features["d_batch_size"], 1], 0.0)	

			self.output_cell = OutputCell(context, out_memory_state, reads, mp_reads)
			output = self.output_cell.forward(args, features)	
	
			# TODO: tidy away later
			self.control_taps = control_taps
			self.read_taps = read_taps
			self.mp_taps = mp_taps
			self.tap_memory_forget = tap_memory_forget

			self.mp_state = out_mp_state
			self.context = context

			
			out_state = (
				out_control_state, 
				out_memory_state, 
				out_mp_state,
			)


			return output, out_state


	def taps(self):

		# TODO: Remove all of this and let it run in the subsystem

		control_taps = self.control_taps
		mp_taps = self.mp_taps
		read_taps = self.read_taps		

		empty_attn = tf.fill([self.features["d_batch_size"], self.features["d_src_len"], 1], 0.0)
		empty_query = tf.fill([self.features["d_batch_size"], self.features["d_src_len"]], 0.0)



		# TODO: AST this all away
		out_taps = {
			"question_word_attn": 		control_taps.get("attn", empty_attn),
			"question_word_attn_raw": 	control_taps.get("attn_raw", empty_attn),
			"mp_node_state":			self.mp_state,
			"iter_id":					self.context.in_iter_id,
		}

		if self.args["use_message_passing"]:

			suffixes = ["_attn", "_attn_raw", "_query", "_signal"]
			for qt in self.args["query_taps"]:
				suffixes.append("_query_"+qt)

			for mp_head in ["mp_write", "mp_read0"]:
				for suffix in suffixes:
					i = mp_head + suffix
					out_taps[i] = mp_taps.get(i, empty_query)

		if self.args["use_read_cell"]:
			
			for j in range(self.args["read_heads"]):

				for i in [f"read{j}_head_attn", f"read{j}_head_attn_focus"]:
					out_taps[i] = read_taps[i]

				if self.args["use_read_previous_outputs"]:
					for i in [f"read{j}_po_content_attn", f"read{j}_po_index_attn"]:
						out_taps[i] = read_taps[i]

				for i in self.args["kb_list"]:
					for k in ["attn", *self.args["query_taps"]]:
						out_taps[f"{i}{j}_{k}"    ] = tf.squeeze(read_taps.get(f"{i}{j}_{k}", empty_attn), 2)
					out_taps[f"{i}{j}_switch_attn"] = read_taps.get(f"{i}{j}_switch_attn", empty_attn)
					out_taps[f"{i}{j}_word_attn"  ] = read_taps.get(f"{i}{j}_word_attn", empty_query)

		

		return out_taps




	def tap_sizes(self):

		# TODO: Merge with taps / remove all this

		# we need a DSL so badly
		def add_query_taps(t, prefix):
			t[f"{prefix}_token_content_attn"] = self.features["d_src_len"]
			t[f"{prefix}_token_index_attn"  ] = self.features["d_src_len"]
			t[f"{prefix}_step_const_signal" ] = self.args["input_width"]
			t[f"{prefix}_memory_attn" 	    ] = self.args["memory_width"] // self.args["input_width"]
			t[f"{prefix}_prev_output_attn"  ] = self.args["max_decode_iterations"]
			t[f"{prefix}_switch_attn" 	    ] = 2


		t = {
			"question_word_attn": 		self.args["control_heads"] * self.features["d_src_len"],
			"question_word_attn_raw": 	self.args["control_heads"] * self.features["d_src_len"],
			"mp_node_state":			tf.TensorShape([self.args["kb_node_max_len"], self.args["mp_state_width"]]),
			"iter_id":					self.args["max_decode_iterations"],
		}

		if self.args["use_message_passing"]:
			for mp_head in ["mp_write", "mp_read0"]:
				t[f"{mp_head}_attn"]			= self.args["kb_node_max_len"]
				t[f"{mp_head}_attn_raw"] 		= self.args["kb_node_max_len"]
				t[f"{mp_head}_query"]			= self.args["kb_node_width"] * self.args["embed_width"]
				t[f"{mp_head}_signal"]			= self.args["mp_state_width"]

				add_query_taps(t, mp_head+"_query")


		if self.args["use_read_cell"]:
			for j in range(self.args["read_heads"]):

				t[f"read{j}_head_attn"        ] = 2 + len(self.args["kb_list"])
				t[f"read{j}_head_attn_focus"  ] = 2 + len(self.args["kb_list"])

				if self.args["use_read_previous_outputs"]:
					t[f"read{j}_po_content_attn"  ] = self.args["max_decode_iterations"]
					t[f"read{j}_po_index_attn"    ] = self.args["max_decode_iterations"]

				for i in self.args["kb_list"]:

					t[f"{i}{j}_attn" 			  ] = self.args[f"{i}_width"] * self.args["embed_width"]
					t[f"{i}{j}_word_attn" 		  ] = self.args[f"{i}_width"]

					add_query_taps(t, f"{i}{j}")

		return t

	def print(self, tap_dict, path):
		pass








