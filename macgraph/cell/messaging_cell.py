
from typing import NamedTuple
import tensorflow as tf

from ..args import ACTIVATION_FNS
from ..attention import *
from ..input import get_table_with_embedding

MP_State = tf.Tensor

class MP_Node(NamedTuple):
	id: str
	properties: tf.Tensor
	state: MP_State

use_message_passing_fn = False
use_self_reference = False


def messaging_cell(args, features, vocab_embedding, in_node_state, in_control_state):
	'''
	Operate a message passing cell
	Each iteration it'll do one round of message passing

	Returns: read_signal, node_state

	for to_node in nodes:
		to_node.state = combine_incoming_signals([
			message_pass(from_node, to_node) for from_node in nodes
		] + [node_self_update(to_node)])  
			

	'''

	node_table, node_table_width, node_table_len = get_table_with_embedding(args, features, vocab_embedding, "kb_node")

	# Read/Write queries
	in_write_query  = tf.layers.dense(in_control_state, node_table_width)
	in_write_signal = tf.layers.dense(in_control_state, args["mp_state_width"])
	in_read_query   = tf.layers.dense(in_control_state, node_table_width)
	
	return do_messaging_cell(args, features, vocab_embedding, 
		in_node_state,
		node_table, node_table_width, node_table_len,
		in_write_query, in_write_signal, in_read_query)



def do_messaging_cell(args, features, vocab_embedding, 
	in_node_state, 
	node_table, node_table_width, node_table_len,
	in_write_query, in_write_signal, in_read_query):

	with tf.name_scope("messaging_cell"):

		node_state = in_node_state
		taps = {}
		taps["mp_write_query"] = in_write_query
		taps["mp_write_signal"] = in_write_signal


		# Add write signal:
		write_signal, taps["mp_write_attn"], _ = attention_write_by_key(
			keys=node_table,
			key_width=node_table_width,
			keys_len=node_table_len,
			query=in_write_query,
			value=in_write_signal,
		)

		delta = tf.shape(node_state)[1] - tf.shape(write_signal)[1]
		write_signal = tf.pad(write_signal, [ [0,0], [0,delta], [0,0] ]) # zero pad out
		write_signal = dynamic_assert_shape(write_signal, tf.shape(node_state), "write_signal")

		node_state += write_signal

		if use_message_passing_fn:
			# Message passing function is a 1d conv [filter_width, in_channels, out_channels]
			message_pass_kernel = tf.get_variable("message_pass_kernel", [1, args["mp_state_width"], args["mp_state_width"]])

			# Apply message pass function:
			node_state = tf.nn.conv1d(node_state, message_pass_kernel, 1, 'SAME', name="message_pass")

		# Aggregate via adjacency matrix with normalisation (that does not include self-edges)
		adj = tf.cast(features["kb_adjacency"], tf.float32)
		degree = tf.reduce_sum(adj, -1, keep_dims=True)
		inv_degree = tf.reciprocal(degree)
		node_mask = tf.expand_dims(tf.sequence_mask(features["kb_nodes_len"], args["kb_node_max_len"]), -1)
		inv_degree = tf.where(node_mask, inv_degree, tf.zeros(tf.shape(inv_degree)))
		inv_degree = tf.where(tf.greater(degree, 0), inv_degree, tf.zeros(tf.shape(inv_degree)))
		inv_degree = tf.check_numerics(inv_degree, "inv_degree")
		adj_norm = inv_degree * adj
		adj_norm = tf.cast(adj_norm, node_state.dtype)
		adj_norm = tf.check_numerics(adj_norm, "adj_norm")
		agg = tf.einsum('bnw,bnm->bmw', node_state, adj_norm)

		if use_self_reference:
			# Add self-reference
			self_reference_kernel = tf.get_variable("message_pass_kernel", [1, args["mp_state_width"], args["mp_state_width"]])
			sr = tf.nn.conv1d(in_node_state, self_reference_kernel, 1, 'SAME', name="self_reference")
			sr *= args["mp_self_dampening"]
			node_state = agg + sr

		# Apply activation
		node_state = ACTIVATION_FNS[args["mp_activation"]](node_state)

		taps["mp_node_state"] = node_state

		# Output
		delta = tf.shape(node_state)[1] - tf.shape(node_table)[1]
		padded_node_table = tf.pad(node_table, [ [0,0], [0,delta], [0,0] ]) # zero pad out

		out_read_signal, taps["mp_read_attn"], _ = attention_key_value(
			keys=padded_node_table,
			keys_len=node_table_len,
			key_width=node_table_width,
			query=in_read_query,
			table=node_state,
			)

		return out_read_signal, node_state, taps


