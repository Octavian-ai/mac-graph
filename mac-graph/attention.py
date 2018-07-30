
from .util import *

from .args import global_args
from .const import EPSILON


def softmax_with_masking(logits, mask, axis):
	assert mask.dtype == tf.bool
	mask = dynamic_assert_shape(mask, tf.shape(logits))
	assert axis < len(logits.shape)

	# For numerical stability shrink the values
	logit_max = tf.reduce_max(tf.boolean_mask(logits, mask))

	# Numerator
	l = tf.exp(logits - logit_max) * tf.cast(mask, logits.dtype)
	
	# Denominator
	d = tf.reduce_sum(l, axis) 
	d = tf.expand_dims(d, axis)

	return l / (d + EPSILON)


def attention(table, query, word_size=None, table_len=None, table_max_len=None):
	"""
	Apply attention

	Arguments:
		- `query`, shape (batch_size, width)
		- `table`, shape (batch_size, len, width)
		- word_size: The width of the table entries
		- `table_len` A tensor of the lengths of the tables (in the batch) that is used to mask the scores before applying softmax (i.e. meaning that any table values after the length are ignored in the lookup)

	"""
	
	with tf.name_scope("attention"):

		q = query
		db = table

		# --------------------------------------------------------------------------
		# Validate inputs
		# --------------------------------------------------------------------------

		assert len(table.shape) == 3, "table should be shape [batch, len, width]"

		batch_size = tf.shape(db)[0]
		seq_len = tf.shape(db)[1]

		if word_size is None:
			word_size = tf.shape(db)[2]

		db_shape = tf.shape(table)
		q_shape = [batch_size, word_size]
		scores_shape = [batch_size, seq_len, 1]

		q = dynamic_assert_shape(q, q_shape)

		# --------------------------------------------------------------------------
		# Run model
		# --------------------------------------------------------------------------

		scores = tf.matmul(db, tf.expand_dims(q, 2))

		if table_len is not None:
			scores_mask = tf.sequence_mask(table_len, seq_len)
			scores_mask = tf.expand_dims(scores_mask, -1) # I like to tightly assert my shapes
			scores_mask = dynamic_assert_shape(scores_mask, scores_shape)
			scores = softmax_with_masking(scores, mask=scores_mask, axis=1)
		else:
			scores = tf.nn.softmax(scores, axis=1)

		scores = dynamic_assert_shape(scores, scores_shape)
		
		weighted_db = db * scores

		output = tf.reduce_sum(weighted_db, 1)
		output = dynamic_assert_shape(output, q_shape)

		return output, scores



# A big effort to do a dense layer on scores (it has a dynamic width)
# if global_args["use_attn_score_dense"]:
# 	need_to_set_shape = scores.shape[1].value is None
# 	assert word_size is not None, "Cannot use_dense with unknown width_size"
# 	assert not need_to_set_shape or table_max_len is not None, f"Please supply max seq len since seq len {db.name} is dynamic"

# 	scores = tf.squeeze(scores, axis=2)

# 	if need_to_set_shape:
# 		delta = table_max_len - seq_len
# 		scores = tf.pad(scores, [[0, 0], [0, delta]])
# 		scores = tf.reshape(scores, [-1, table_max_len])
		
# 	scores = tf.layers.dense(scores, word_size)

# 	if need_to_set_shape:
# 		scores = scores[:,0:seq_len]

# 	scores = tf.expand_dims(scores, axis=2)
# 	scores = dynamic_assert_shape(scores, scores_shape)
		

