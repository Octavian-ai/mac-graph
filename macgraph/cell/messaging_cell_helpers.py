

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
	

def mp_matmul(state, mat, name):
	return tf.nn.conv1d(state, mat, 1, 'VALID', name=name)


"""
	Where length is the second dimension
"""
def pad_to_table_len(tensor, table_to_mimic, name=None):
	delta = tf.shape(table_to_mimic)[1] - tf.shape(tensor)[1]
	tensor = tf.pad(tensor, [ [0,0], [0,delta], [0,0] ]) # zero pad out
	# tensor = dynamic_assert_shape(tensor, tf.shape(table_to_mimic)[0:1]+[tf.shape(tensor)[2]], name)
	return tensor



def calc_right_shift(node_incoming):
	shape = tf.shape(node_incoming)
	node_incoming = tf.concat([node_incoming[:,:,1:],node_incoming[:,:,0:1]], axis=-1) 
	node_incoming = dynamic_assert_shape(node_incoming, shape, "node_incoming")
	return node_incoming