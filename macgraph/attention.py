
import tensorflow as tf

from .util import *

from .args import global_args
from .const import EPSILON


def softmax_with_masking(logits, mask, axis, name="", internal_dtype=tf.float64):
	with tf.name_scope(name+"_softmax_with_masking"):

		# --------------------------------------------------------------------------
		# Validate inputs
		# --------------------------------------------------------------------------
		
		logits_shape = tf.shape(logits)

		assert mask.dtype == tf.bool
		mask = dynamic_assert_shape(mask, logits_shape)
		assert axis < len(logits.shape)
		logits = tf.check_numerics(logits, "logits")
		mask = dynamic_assert_shape(mask, logits_shape, "mask")

		# --------------------------------------------------------------------------
		# Mask those logits!
		# --------------------------------------------------------------------------

		# masked_logits = tf.boolean_mask(logits, mask)
		# masked_logits = tf.reshape(masked_logits, tf.shape(logits))

		f_mask = tf.cast(mask, internal_dtype)
		masked_logits = tf.cast(logits, internal_dtype) * f_mask
		masked_logits = dynamic_assert_shape(masked_logits, logits_shape, "masked_logits")

		# masked_logits = tf.Print(masked_logits, [f"{name}: masked_logits", tf.squeeze(masked_logits)], message="\n", summarize=9999)

		# For numerical stability shrink the values
		logits_max = tf.reduce_max(masked_logits, axis=axis, keepdims=True)
		logits_max = tf.check_numerics(logits_max, "logit_max")

		# Numerator
		l_delta = (tf.cast(logits, internal_dtype) - logits_max) * f_mask
		l_delta = tf.check_numerics(l_delta, "l_delta")
		l_delta = dynamic_assert_shape(l_delta, tf.shape(logits), "l_delta")

	
		# l_delta = tf.Print(l_delta, [f"{name}: logits", tf.squeeze(logits,-1)], message="\n", summarize=9999)
		# l_delta = tf.Print(l_delta, [f"{name}: logits_max", logits_max], message="\n", summarize=9999)
		# l_delta = tf.Print(l_delta, [f"{name}: l_delta", tf.squeeze(l_delta,-1)], message="\n", summarize=9999)

		# This assert fails, howwwww??
		with tf.control_dependencies([tf.assert_less_equal(l_delta, tf.cast(0.0, l_delta.dtype), summarize=100000, data=[logits_max, mask, logits])]):
			
			l = tf.exp(l_delta)
			l = tf.check_numerics(l, "numerator pre mask")

			# l = tf.Print(l, [f"numerator pre mask {name}", tf.squeeze(l,-1)], message="\n", summarize=9999)

			l *= f_mask
			l = tf.check_numerics(l, "numerator")
			# l = tf.Print(l, [f"numerator post mask {name}", tf.squeeze(l,-1)], message="\n", summarize=9999)
			
			# Denominator
			d = tf.reduce_sum(l, axis) 
			d = tf.expand_dims(d, axis)
			d = tf.check_numerics(d, "denominator")

			normalized = l / (d + EPSILON)
			normalized = tf.cast(normalized, logits.dtype)

			normalized = dynamic_assert_shape(normalized, logits_shape, "normalized_sm_scores")

			# Total, by batch
			scores_total = tf.reduce_sum(normalized, axis=axis)
			# keys_more_than_zero = tf.where(
			# 	tf.greater(keys_len, 0),
			# 	tf.ones(tf.shape(scores_total)), tf.zeros(tf.shape(scores_total)))

			sum_to_one = tf_assert_almost_equal(scores_total, 1.0, message=f"Checking scores sum to 1.0",summarize=999)
			
			with tf.control_dependencies([sum_to_one]):
				return normalized


def attention(table:tf.Tensor, query:tf.Tensor, key_width:int=None, keys_len=None, name="attention"):
	"""
	Returns:
		- attention_output
		- focus
		- taps {"attn", "attn_raw"}
	"""

	return attention_key_value(
		keys=table, 
		table=table, 
		query=query, 
		key_width=key_width, keys_len=keys_len, name=name)

def attention_key_value(keys:tf.Tensor, table:tf.Tensor, query:tf.Tensor, key_width:int=None, keys_len=None, name="attention"):
	"""
	Apply attention

	Arguments:
		- `keys`, shape (batch_size, len, key_width)
		- `query`, shape (batch_size, key_width)
		- `table`, shape (batch_size, len, value_width)
		- key_width: The width of the key entries
		- `keys_len` A tensor of the lengths of the tables (in the batch) that is used to mask the scores before applying softmax (i.e. meaning that any table values after the length are ignored in the lookup)


	Returns:
		- attention_output
		- focus
		- taps {"attn", "attn_raw"}
	"""

	assert len(table.shape) == 3, "table should be shape [batch, seq_len, value_width]"
	batch_size = tf.shape(table)[0]
	seq_len = tf.shape(table)[1]
	value_width = tf.shape(table)[2]

	keys = dynamic_assert_shape(keys, [batch_size, seq_len, tf.shape(keys)[2]], "keys")

	scores_sm, attn_focus, scores_raw = attention_compute_scores(
		keys=keys, 
		query=query, 
		key_width=key_width, 
		keys_len=keys_len, 
		name=name)

	scores_sm = dynamic_assert_shape(scores_sm, [batch_size, seq_len, 1], "scores_sm")

	with tf.name_scope(name):
		weighted_table = table * scores_sm

		output = tf.reduce_sum(weighted_table, 1)
		output = dynamic_assert_shape(output, [batch_size, value_width], "output")
		output = tf.check_numerics(output, "attention_output")

		return output, attn_focus, {"attn": scores_sm, "attn_raw": scores_raw}

def attention_compute_scores(keys:tf.Tensor, query:tf.Tensor, key_width:int=None, keys_len=None, name:str="attention"):
	with tf.name_scope(name):

		# --------------------------------------------------------------------------
		# Validate inputs
		# --------------------------------------------------------------------------

		assert query is not None
		assert keys is not None
		assert len(keys.shape) == 3, "keys should be shape [batch, len, key_width]"

		batch_size = tf.shape(keys)[0]
		seq_len = tf.shape(keys)[1]

		if keys_len is not None:
			keys_len = dynamic_assert_shape(keys_len, [batch_size], "keys_len")

		if key_width is None:
			key_width = tf.shape(keys)[2]

		q_shape = [batch_size, key_width]
		scores_shape = [batch_size, seq_len, 1]
		keys_shape = [batch_size, seq_len, key_width]
		
		query = dynamic_assert_shape(query, q_shape, "query")
		keys = dynamic_assert_shape(keys, keys_shape, "keys") # Somewhat tautalagous

		# --------------------------------------------------------------------------
		# Run model
		# --------------------------------------------------------------------------

		scores = tf.matmul(keys, tf.expand_dims(query, 2))
		scores = dynamic_assert_shape(scores, scores_shape, "scores")

		if keys_len is not None:
			scores_mask = tf.sequence_mask(keys_len, seq_len)
			scores_mask = tf.expand_dims(scores_mask, -1)
			scores_mask = dynamic_assert_shape(scores_mask, scores_shape, "scores_mask")
			scores_sm = softmax_with_masking(scores, mask=scores_mask, axis=1, name=name)
		else:
			scores_sm = tf.nn.softmax(scores + EPSILON, axis=1)
			
		scores_sm = dynamic_assert_shape(scores_sm, scores_shape, "scores_sm")

		return scores_sm, tf.reduce_sum(scores, axis=1), scores


def attention_write_by_key(keys, query, value, key_width=None, keys_len=None, name="attention"):
	"""
	Returns:
		- attention_output
		- softmax_scores
		- focus
	"""
	
	batch_size = tf.shape(keys)[0]
	seq_len = tf.shape(keys)[1]
	value_width = tf.shape(value)[-1]

	assert len(value.shape) == 2, "Value must have batch dimension"

	scores_sm, attn_focus, scores_raw = attention_compute_scores(
		keys=keys, query=query, key_width=key_width, keys_len=keys_len, name=name)

	with tf.name_scope(name):
		weighted_table = tf.expand_dims(value, 1) * scores_sm	
		weighted_table = dynamic_assert_shape(weighted_table, [batch_size, seq_len, value_width])
		return weighted_table, attn_focus, {"attn": scores_sm, "attn_raw": scores_raw}





def attention_by_index(control, head_stack, name:str="attention_by_index"):
	'''
	Essentially a weighted sum over the second-last dimension of head_stack, 
	using a dense softmax of control for the weights


	Shapes:
		* control [batch, word_size]
		* head_stack [batch, seq_len, word_size]

	Returns [batch, word_size]		

	'''

	with tf.name_scope(name):

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
		

