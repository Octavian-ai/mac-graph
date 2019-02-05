
from typing import NamedTuple
import tensorflow as tf

from .types import *
from .query import *
from .messaging_cell_helpers import *

from ..args import ACTIVATION_FNS
from ..attention import *
from ..input import get_table_with_embedding
from ..const import EPSILON
from ..util import *
from ..layers import *
from ..activations import *


def lerp (a, b, f):
	return (1-f)*a + f*b

class MessagingCell(Component):

	def __init__(self, args):
		super().__init__(args, name="mp")

		self.question_tokens = Tensor("question_tokens")
		self.read_gs_attn = AttentionByIndex(args, 
			table=self.question_tokens,
			seq_len=args["max_seq_len"],
			table_representation="src", name="read_gs_attn")

	def forward(self, features, context):
	
		node_table, node_table_width, node_table_len = get_table_with_embedding(context.args, context.features, context.vocab_embedding, "kb_node")

		in_signal = tf.concat([context.control_state, context.in_iter_id], -1)

		control_parts = tf.reshape(context.control_state, [context.features["d_batch_size"], -1, context.args["input_width"]])

		taps = {}
		def add_taps(val, prefix):
			ret,tps = val
			for k,v in tps.items():
				taps[prefix+"_"+k] = v
			return ret


		in_write_signal 		= layer_dense(in_signal, context.args["mp_state_width"], "sigmoid")
		# in_write_signal			= tf.ones([context.features["d_batch_size"], context.args["mp_state_width"]])


		# Read/Write queries
		# in_write_query 		= context.control_state
		# in_write_query		= layer_dense(context.control_state, node_table_width)
		# in_write_query  		= context.in_question_tokens[:,10,:]
		in_write_query			= add_taps(generate_token_index_query(context, "write_query"), "write_query")
		# in_read0_query			= context.in_question_tokens[:,14,:] 
		# in_read0_query 			= control_parts[:,0,:]
		# in_read0_query			= tf.layers.dense(generate_query(context, "mp_read_query")[0], node_table_width)
		

		read_queries = []
		for i in range(context.args["mp_read_heads"]):
			read_queries.append(add_taps(generate_token_index_query(context, f"read{i}_query"), f"read{i}_query"))
		
		# self.question_tokens.bind(context.in_question_tokens_padded)
		# global_signal = self.read_gs_attn.forward(features)
		global_signal = context.in_question_tokens[:,26,:] # just the cleanliness signal

		out_read_signals, node_state, taps2 = self.do_messaging_cell(context,
			node_table, node_table_width, node_table_len,
			in_write_query, in_write_signal, read_queries, global_signal)

		self._taps = {**taps, **taps2}


		return out_read_signals, node_state


	def taps(self):
		return self._taps

	def tap_sizes(self):

		t = {}

		mp_reads = [f"read{i}" for i in range(self.args["mp_read_heads"])]

		for mp_head in ["write", *mp_reads]:
			t[f"{mp_head}_attn"]			= self.args["kb_node_max_len"]
			t[f"{mp_head}_attn_raw"] 		= self.args["kb_node_max_len"]
			t[f"{mp_head}_query"]			= self.args["kb_node_width"] * self.args["embed_width"]
			t[f"{mp_head}_signal"]			= self.args["mp_state_width"]
			t[f"{mp_head}_query_token_index_attn"  ] = self.args["max_seq_len"]

		return t









	def do_messaging_cell(self, context:CellContext, 
		node_table, node_table_width, node_table_len,
		in_write_query, in_write_signal, in_read_queries, global_signal):

		'''
		Operate a message passing cell
		Each iteration it'll do one round of message passing

		Returns: read_signal, node_state

		for to_node in nodes:
			to_node.state = combine_incoming_signals([
				message_pass(from_node, to_node) for from_node in to_node.neighbors
			] + [node_self_update(to_node)])  
				

		'''

		with tf.name_scope("messaging_cell"):

			taps = {}
			taps["write_query"] = in_write_query
			taps["write_signal"] = in_write_signal

			node_state_shape = tf.shape(context.in_node_state)
			node_state = context.in_node_state
			padded_node_table = pad_to_table_len(node_table, node_state, "padded_node_table")

			node_ids_width = self.args["embed_width"]
			node_ids = node_table[:,:,0:node_ids_width]
			padded_node_ids  =padded_node_table[:,:,0:node_ids_width]
			node_ids_len = node_table_len

			# --------------------------------------------------------------------------
			# Write to graph
			# --------------------------------------------------------------------------
			
			write_signal, _, a_taps = attention_write_by_key(
				keys     =node_ids,
				key_width=node_ids_width,
				keys_len =node_ids_len,
				query=in_write_query,
				value=in_write_signal,
				name="write_signal"
			)
			for k,v in a_taps.items():
				taps["write_"+k] = v

			write_signal = pad_to_table_len(write_signal, node_state, "write_signal")
			node_state += write_signal
			node_state = dynamic_assert_shape(node_state, node_state_shape, "node_state")
			
			# --------------------------------------------------------------------------
			# Calculate adjacency 
			# --------------------------------------------------------------------------

			node_incoming = calc_normalized_adjacency(context, node_state)

			if context.args["use_mp_right_shift"]:
				node_incoming = calc_right_shift(node_incoming)


			# --------------------------------------------------------------------------
			# Perform propagation
			# --------------------------------------------------------------------------
			
			if context.args["use_mp_gru"]:
				node_state = self.node_cell(context, node_state, node_incoming, padded_node_table, global_signal)

			else:
				node_state = node_incoming

			# --------------------------------------------------------------------------
			# Read from graph
			# --------------------------------------------------------------------------

			out_read_signals = []

			for idx, qry in enumerate(in_read_queries):
				out_read_signal, _, a_taps = attention_key_value(
					keys     =padded_node_ids,
					keys_len =node_ids_len,
					key_width=node_ids_width,
					query=qry,
					table=node_state,
					name=f"read{idx}"
					)
				out_read_signals.append(out_read_signal)

				for k,v in a_taps.items():
					taps[f"read{idx}_{k}"] = v
				taps[f"read{idx}_signal"] = out_read_signal
				taps[f"read{idx}_query"] = qry


			taps["node_state"] = node_state
			node_state = dynamic_assert_shape(node_state, node_state_shape, "node_state")
			assert node_state.shape[-1] == context.in_node_state.shape[-1], "Node state should not lose dimension"

			return out_read_signals, node_state, taps



	def node_cell(self, context, node_state, node_incoming, padded_node_table, global_signal):

		# --------------------------------------------------------------------------
		# Sizes
		# --------------------------------------------------------------------------
		
		seq_len = padded_node_table.shape[1]
		n_features = self.args["kb_node_width"]
		feature_width = self.args["embed_width"]

		# --------------------------------------------------------------------------
		# Global signal comparison
		# --------------------------------------------------------------------------
		
		node_properties = tf.reshape(padded_node_table, 
			[context.features["d_batch_size"], seq_len, n_features, feature_width])

		node_cleanliness = node_properties[:,:,1,:]
		# node_cleanliness = node_dense(node_cleanliness, feature_width, activation="selu", name="node_cleanliness")

		t_global_signal = layer_dense(global_signal, feature_width, "selu")
		node_cleanliness_tgt = tf.expand_dims(t_global_signal, 1)
		
		node_cleanliness_score = tf.reduce_sum(node_cleanliness * node_cleanliness_tgt, axis=2, keepdims=True)

		node_cleanliness_score = dynamic_assert_shape(node_cleanliness_score, 
			[context.features["d_batch_size"], seq_len, 1])


		# --------------------------------------------------------------------------
		# RNN Cell
		# --------------------------------------------------------------------------
		
		all_inputs = [node_state, node_incoming]
		# all_inputs.append(padded_node_table)
		# all_inputs.append(tf.tile(tf.expand_dims(global_signal,1), [1, node_state.shape[1], 1]))
		# all_inputs.append(node_cleanliness_score)
		all_inputs = tf.concat(all_inputs, axis=-1)


		signals = {}
		for s in ["forget"]:
			signals[s] = node_dense(all_inputs, context.args["mp_state_width"], activation="sigmoid", name=s+"_signal")
			
			if self.args["use_summary_scalar"]:
				tf.summary.histogram("mp_"+s, signals[s])
			
		out_node_state = node_incoming * signals["forget"]

		return out_node_state





