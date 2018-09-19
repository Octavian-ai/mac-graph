
import tensorflow as tf

from .util import *

from .args import global_args
from .const import EPSILON


def softmax_with_masking(logits, mask, axis):
	with tf.name_scope("softmax_with_masking"):
		assert mask.dtype == tf.bool
		mask = dynamic_assert_shape(mask, tf.shape(logits))
		assert axis < len(logits.shape)
		logits = tf.check_numerics(logits, "logits")

		# For numerical stability shrink the values
		logits_max = tf.reduce_max(tf.boolean_mask(logits, mask))
		logits_max = tf.check_numerics(logits_max, "logit_max")

		f_mask = tf.cast(mask, logits.dtype)

		# Numerator
		l_delta = (logits - logits_max) * f_mask
		l_delta = tf.check_numerics(l_delta, "l_delta")

		# This assert fails, howwwww??
		with tf.control_dependencies([tf.assert_less_equal(l_delta, 0.0, summarize=100000, data=[logits_max, mask, logits])]):
			
			l = tf.exp(l_delta)
			l = tf.check_numerics(l, "numerator pre mask")
			l *= tf.cast(mask, logits.dtype)
			l = tf.check_numerics(l, "numerator")
			
			# Denominator
			d = tf.reduce_sum(l, axis) 
			d = tf.expand_dims(d, axis)
			d = tf.check_numerics(d, "denominator")

			return l / (d + EPSILON)


def attention(table, query, key_width=None, keys_len=None, name="attention"):
	return attention_key_value(table, table, query, key_width, keys_len, name)

def attention_key_value(keys, table, query, key_width=None, keys_len=None, name="attention"):
	"""
	Apply attention

	Arguments:
		- `keys`, shape (batch_size, len, key_width)
		- `query`, shape (batch_size, key_width)
		- `table`, shape (batch_size, len, value_width)
		- key_width: The width of the key entries
		- `keys_len` A tensor of the lengths of the tables (in the batch) that is used to mask the scores before applying softmax (i.e. meaning that any table values after the length are ignored in the lookup)

	"""
	
	scores_sm, attn_focus = attention_compute_scores(keys, key_width, keys_len, query, name)

	with tf.name_scope(name):

		assert len(table.shape) == 3, "table should be shape [batch, len, value_width]"
		value_width = tf.shape(table)[2]
		table_shape = tf.shape(table)

		weighted_table = table * scores_sm

		output = tf.reduce_sum(weighted_table, 1)
		output = dynamic_assert_shape(output, [batch_size, value_width], "output")
		output = tf.check_numerics(output, "attention_output")

		return output, scores_sm, attn_focus

def attention_compute_scores(keys, query, key_width=None, keys_len=None, name="attention"):
	with tf.name_scope(name):

		# --------------------------------------------------------------------------
		# Validate inputs
		# --------------------------------------------------------------------------

		assert query is not None
		assert keys is not None
		assert len(keys.shape) == 3, "keys should be shape [batch, len, key_width]"
		
		batch_size = tf.shape(keys)[0]
		seq_len = tf.shape(keys)[1]

		if key_width is None:
			key_width = tf.shape(keys)[2]

		q_shape = [batch_size, key_width]
		scores_shape = [batch_size, seq_len, 1]

		query = dynamic_assert_shape(query, q_shape, "query")

		# --------------------------------------------------------------------------
		# Run model
		# --------------------------------------------------------------------------

		scores = tf.matmul(keys, tf.expand_dims(query, 2))

		if keys_len is not None:
			scores_mask = tf.sequence_mask(keys_len, seq_len)
			scores_mask = tf.expand_dims(scores_mask, -1)
			scores_mask = dynamic_assert_shape(scores_mask, scores_shape, "scores_mask")
			scores_sm = softmax_with_masking(scores, mask=scores_mask, axis=1)
		else:
			scores_sm = tf.nn.softmax(scores + EPSILON, axis=1)

		scores_sm = dynamic_assert_shape(scores_sm, scores_shape, "scores")

		return scores_sm, tf.reduce_sum(scores, axis=1)


def attention_write_by_key(keys, query, value, key_width=None, keys_len=None, name="attention"):

	batch_size = tf.shape(keys)[0]
	seq_len = tf.shape(keys)[1]
	value_width = tf.shape(value)[-1]

	assert len(value.shape) == 2, "Value must have batch dimension"

	scores_sm, attn_focus = attention_compute_scores(keys, query, key_width, keys_len, name)

	with tf.name_scope(name):
		weighted_table = tf.expand_dims(value, 1) * scores_sm	
		weighted_table = dynamic_assert_shape(weighted_table, [batch_size, seq_len, value_width])
		return weighted_table, scores_sm, attn_focus





def attention_by_index(control, head_stack):
	'''
	Essentially a weighted sum over the second-last dimension of head_stack, 
	using a dense softmax of control for the weights


	Shapes:
		* control [batch, word_size]
		* head_stack [batch, seq_len, word_size]

	Returns [batch, word_size]		

	'''

	with tf.name_scope("attention_by_index"):

		word_size = tf.shape(head_stack)[-1]
		seq_len = head_stack.shape[-2]
		output_shape = [tf.shape(control)[0], word_size]

		assert seq_len is not None, "Seq len must be defined"
		
		query = tf.layers.dense(control, seq_len, activation=tf.nn.softmax)
		
		weighted_stack = head_stack * tf.expand_dims(query, -1)
		weighted_sum = tf.reduce_sum(weighted_stack, -2)

		output = weighted_sum
		output = dynamic_assert_shape(output, output_shape)
		return output, query
		

