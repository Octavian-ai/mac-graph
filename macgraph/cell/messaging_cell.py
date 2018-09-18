
from typing import NamedTuple
import tensorflow as tf

from ..args import ACTIVATION_FNS

MP_State = tf.Tensor

class MP_Node(NamedTuple):
	id: str
	properties: tf.Tensor
	state: MP_State



def messaging_cell(args, features, in_node_state, in_write_query, in_write_signal, in_read_query):
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
	write_signal = attention_write_by_key(
		keys=features["kb_nodes"],
		query=in_write_query,
		signal=in_write_signal,
	)

	node_state += write_signal

	# Message passing function is a 1d conv [filter_width, in_channels, out_channels]
	message_pass_kernel = tf.get_variable("message_pass_kernel", [1, args["kb_node_state_width"], args["kb_node_state_width"]])

	# Apply message pass function:
	node_state = tf.nn.conv1d(node_state, message_pass_kernel, 1, 'SAME', name="message_pass")

	# Aggregate via adjacency matrix (that does not include self-edges)
	agg = tf.matmul(node_state, features["kb_node_adjacency"])

	# Add self-reference
	self_reference_kernel = tf.get_variable("message_pass_kernel", [1, args["mp_state_width"], args["kb_node_state_width"]])
	sr = tf.nn.conv1d(in_node_state, self_reference_kernel, 1, 'SAME', name="self_reference")
	sr *= args["mp_self_dampening"]
	node_state = agg + sr

	# Apply activation
	node_state = ACTIVATION_FNS[args["mp_activation"]](node_state)


	# Outputs
	out_node_state = node_state

	out_read_signal = attention_key_value(
		table=out_node_state,
		keys=features["kb_nodes"],
		query=in_read_query,
		word_size=args["kb_node_width"])


	return out_read_signal, out_node_state