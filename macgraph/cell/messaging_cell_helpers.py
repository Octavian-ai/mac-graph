

from typing import NamedTuple
import tensorflow as tf

from .types import *
from .query import *

from ..args import ACTIVATION_FNS
from ..attention import *
from ..input import get_table_with_embedding
from ..const import EPSILON
from ..util import *
from ..layers import *
from ..activations import *

MP_State = tf.Tensor

class MP_Node(NamedTuple):
	id: str
	properties: tf.Tensor
	state: MP_State

use_message_passing_fn = False
use_self_reference = False

def layer_normalize(tensor):
	'''Apologies if I've abused this term'''

	in_shape = tf.shape(tensor)
	axes = list(range(1, len(tensor.shape)))

	# Keep batch axis
	t = tf.reduce_sum(tensor, axis=axes )
	t += EPSILON
	t = tf.reciprocal(t)
	t = tf.check_numerics(t, "1/sum")

	tensor = tf.einsum('brc,b->brc', tensor, t)

	tensor = dynamic_assert_shape(tensor, in_shape, "layer_normalize_tensor")
	return tensor
	


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

def mp_matmul(state, mat, name):
	return tf.nn.conv1d(state, mat, 1, 'VALID', name=name)





def calc_right_shift(node_incoming):
	shape = tf.shape(node_incoming)
	node_incoming = tf.concat([node_incoming[:,:,1:],node_incoming[:,:,0:1]], axis=-1) 
	node_incoming = dynamic_assert_shape(node_incoming, shape, "node_incoming")
	return node_incoming