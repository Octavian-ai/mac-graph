
from typing import NamedTuple
import tensorflow as tf

from ..args import ACTIVATION_FNS
from ..attention import *

MP_State = tf.Tensor

class MP_Node(NamedTuple):
	id: str
	properties: tf.Tensor
	state: MP_State


def messaging_cell(args, features, in_node_state, in_control_state):

	flat_control_state = tf.reshape(in_control_state, 
		[features["d_batch_size"], args["control_width"] * args["control_heads"]])

	in_write_query  = tf.layers.dense(flat_control_state, args["kb_node_width"])
	in_write_signal = tf.layers.dense(flat_control_state, args["mp_state_width"])
	in_read_query   = tf.layers.dense(flat_control_state, args["kb_node_width"])

	return do_message_pass(args, features, in_node_state, in_write_query, in_write_signal, in_read_query)



def do_message_pass(args, features, in_node_state, in_write_query, in_write_signal, in_read_query):
	'''
	Operate a message passing cell
	Each iteration it'll do one round of message passing

	Returns: read_signal, node_state

	for to_node in nodes:
		to_node.state = combine_incoming_signals([
			message_pass(from_node, to_node) for from_node in nodes
		] + [node_self_update(to_node)])  
			

	'''

	node_state = in_node_state

	# Add write signal:
	write_signal, _, _ = attention_write_by_key(
		keys=features["kb_nodes"],
		key_width=args["kb_node_width"],
		keys_len=features["kb_nodes_len"],
		query=in_write_query,
		value=in_write_signal,
	)

	node_state += write_signal

	# Message passing function is a 1d conv [filter_width, in_channels, out_channels]
	message_pass_kernel = tf.get_variable("message_pass_kernel", [1, args["kb_node_state_width"], args["kb_node_state_width"]])

	# Apply message pass function:
	node_state = tf.nn.conv1d(node_state, message_pass_kernel, 1, 'SAME', name="message_pass")

	# Aggregate via adjacency matrix (that does not include self-edges)
	agg = tf.matmul(node_state, features["kb_adjacency"])

	# Add self-reference
	self_reference_kernel = tf.get_variable("message_pass_kernel", [1, args["mp_state_width"], args["kb_node_state_width"]])
	sr = tf.nn.conv1d(in_node_state, self_reference_kernel, 1, 'SAME', name="self_reference")
	sr *= args["mp_self_dampening"]
	node_state = agg + sr

	# Apply activation
	node_state = ACTIVATION_FNS[args["mp_activation"]](node_state)


	# Outputs
	out_node_state = node_state

	out_read_signal, _, _ = attention_key_value(
		keys=features["kb_nodes"],
		keys_len=feature["kb_nodes_len"],
		key_width=args["kb_node_width"],
		query=in_read_query,
		table=out_node_state,
		)


	return out_read_signal, out_node_state