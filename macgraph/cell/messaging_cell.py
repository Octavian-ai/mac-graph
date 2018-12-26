
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


def messaging_cell(context:CellContext):

	node_table, node_table_width, node_table_len = get_table_with_embedding(context.args, context.features, context.vocab_embedding, "kb_node")

	node_table_width = context.args["input_width"]
	node_table = node_table[:,:,0:node_table_width]

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
	in_write_query			= add_taps(generate_token_index_query(context, "mp_write_query"), "mp_write_query")
	# in_read0_query			= context.in_question_tokens[:,14,:] 
	# in_read0_query 			= control_parts[:,0,:]
	# in_read0_query			= tf.layers.dense(generate_query(context, "mp_read_query")[0], node_table_width)
	

	read_queries = []
	for i in range(context.args["mp_read_heads"]):
		read_queries.append(add_taps(generate_token_index_query(context, f"mp_read{i}_query"), f"mp_read{i}_query"))
	
	out_read_signals, node_state, taps2 = do_messaging_cell(context,
		node_table, node_table_width, node_table_len,
		in_write_query, in_write_signal, read_queries)


	return out_read_signals, node_state, {**taps, **taps2}





def calc_normalized_adjacency(context, node_state):
	# Aggregate via adjacency matrix with normalisation (that does not include self-edges)
	adj = tf.cast(context.features["kb_adjacency"], tf.float32)
	degree = tf.reduce_sum(adj, -1, keepdims=True)
	inv_degree = tf.reciprocal(degree)
	node_mask = tf.expand_dims(tf.sequence_mask(context.features["kb_nodes_len"], context.args["kb_node_max_len"]), -1)
	inv_degree = tf.where(node_mask, inv_degree, tf.zeros(tf.shape(inv_degree)))
	inv_degree = tf.where(tf.greater(degree, 0), inv_degree, tf.zeros(tf.shape(inv_degree)))
	inv_degree = tf.check_numerics(inv_degree, "inv_degree")
	adj_norm = inv_degree * adj
	adj_norm = tf.cast(adj_norm, node_state.dtype)
	adj_norm = tf.check_numerics(adj_norm, "adj_norm")
	node_incoming = tf.einsum('bnw,bnm->bmw', node_state, adj_norm)

	return node_incoming


def node_investiage_gru(context, node_state, node_incoming, padded_node_table):

	all_inputs = [node_state, node_incoming]

	old_and_new = tf.concat(all_inputs, axis=-1)

	input_width = old_and_new.shape[-1]

	forget_w     = tf.get_variable("mp_forget_w",    [1, input_width, context.args["mp_state_width"]], 	initializer=tf.initializers.random_uniform)
	forget_b     = tf.get_variable("mp_forget_b",    [1, context.args["mp_state_width"]],				initializer=tf.initializers.random_uniform)
	reuse_w      = tf.get_variable("mp_reuse_w",     [1, input_width, context.args["mp_state_width"]], 	initializer=tf.initializers.random_uniform)

	transform_w  = tf.get_variable("mp_transform_w", [1, 2 * context.args["mp_state_width"], context.args["mp_state_width"]], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0))
	transform_b  = tf.get_variable("mp_transform_b", [1, context.args["mp_state_width"]], 				initializer=tf.initializers.random_uniform)


	# Initially likely to be zero
	forget_signal = tf.nn.sigmoid(mp_matmul(old_and_new , forget_w, 'forget_signal') + forget_b)
	# reuse_signal  = tf.nn.sigmoid(mp_matmul(old_and_new , reuse_w,  'reuse_signal'))

	# reuse_and_new = tf.concat([reuse_signal * node_state, node_incoming], axis=-1)
	# proposed_new_state = ACTIVATION_FNS[context.args["mp_activation"]](mp_matmul(reuse_and_new, transform_w, 'proposed_new_state'))

	transformed = mp_matmul(old_and_new, transform_w, 'proposed_new_state') + transform_b

	proposed_new_state = ACTIVATION_FNS[context.args["mp_activation"]](node_incoming)

	node_state = (1-forget_signal) * node_state + (forget_signal) * proposed_new_state

	return node_state


def node_gru(context, node_state, node_incoming, padded_node_table):

	all_inputs = [node_state, node_incoming]

	if context.args["use_mp_node_id"]:
		all_inputs.append(padded_node_table[:,:,:context.args["embed_width"]])

	old_and_new = tf.concat(all_inputs, axis=-1)

	input_width = old_and_new.shape[-1]

	forget_w     = tf.get_variable("mp_forget_w",    [1, input_width, context.args["mp_state_width"]])
	forget_b     = tf.get_variable("mp_forget_b",    [1, context.args["mp_state_width"]])

	reuse_w      = tf.get_variable("mp_reuse_w",     [1, input_width, context.args["mp_state_width"]])
	transform_w  = tf.get_variable("mp_transform_w", [1, 2 * context.args["mp_state_width"], context.args["mp_state_width"]])

	# Initially likely to be zero
	forget_signal = tf.nn.sigmoid(mp_matmul(old_and_new , forget_w, 'forget_signal') + forget_b)
	reuse_signal  = tf.nn.sigmoid(mp_matmul(old_and_new , reuse_w,  'reuse_signal'))

	reuse_and_new = tf.concat([reuse_signal * node_state, node_incoming], axis=-1)
	proposed_new_state = ACTIVATION_FNS[context.args["mp_activation"]](mp_matmul(reuse_and_new, transform_w, 'proposed_new_state'))

	node_state = (1-forget_signal) * node_state + (forget_signal) * proposed_new_state

	return node_state



def do_messaging_cell(context:CellContext, 
	node_table, node_table_width, node_table_len,
	in_write_query, in_write_signal, in_read_queries):

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
		taps["mp_write_query"] = in_write_query
		taps["mp_write_signal"] = in_write_signal

		node_state_shape = tf.shape(context.in_node_state)
		node_state = context.in_node_state
		padded_node_table = pad_to_table_len(node_table, node_state, "padded_node_table")
		
		# --------------------------------------------------------------------------
		# Write to graph
		# --------------------------------------------------------------------------
		
		write_signal, _, a_taps = attention_write_by_key(
			keys=node_table,
			key_width=node_table_width,
			keys_len=node_table_len,
			query=in_write_query,
			value=in_write_signal,
			name="mp_write_signal"
		)
		for k,v in a_taps.items():
			taps["mp_write_"+k] = v

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
			node_state = node_investiage_gru(context, node_state, node_incoming, padded_node_table)

		else:
			node_state = node_incoming

		# --------------------------------------------------------------------------
		# Read from graph
		# --------------------------------------------------------------------------

		out_read_signals = []

		for idx, qry in enumerate(in_read_queries):
			out_read_signal, _, a_taps = attention_key_value(
				keys=padded_node_table,
				keys_len=node_table_len,
				key_width=node_table_width,
				query=qry,
				table=node_state,
				name=f"mp_read{idx}"
				)
			out_read_signals.append(out_read_signal)

			for k,v in a_taps.items():
				taps[f"mp_read{idx}_{k}"] = v
			taps[f"mp_read{idx}_signal"] = out_read_signal
			taps[f"mp_read{idx}_query"] = qry


		taps["mp_node_state"] = node_state
		node_state = dynamic_assert_shape(node_state, node_state_shape, "node_state")
		assert node_state.shape[-1] == context.in_node_state.shape[-1], "Node state should not lose dimension"

		return out_read_signals, node_state, taps


