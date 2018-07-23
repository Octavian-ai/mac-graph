
from .util import *


def attention(database, query, mask=None, word_size=None, use_dense=True, output_taps=False, max_len=None):
	"""
	Apply attention

	Arguments:
		- `query` shape (batch_size, width)
		- `database` shape (batch_size, len, width)
		- Optional `mask` shape (batch_size, width)
		- use_dense Whether to instantiate a free variable for comparison function

	"""
	
	with tf.name_scope("attention"):

		q = query
		db = database

		# --------------------------------------------------------------------------
		# Validate inputs
		# --------------------------------------------------------------------------

		assert len(database.shape) == 3, "Database should be shape [batch, len, width]"

		batch_size = tf.shape(db)[0]
		seq_len = tf.shape(db)[1]

		if word_size is None:
			word_size = tf.shape(db)[2]

		db_shape = tf.shape(database)
		q_shape = [batch_size, word_size]
		scores_shape = [batch_size, seq_len, 1]


		q = dynamic_assert_shape(q, q_shape)

		if mask is not None:
			mask = dynamic_assert_shape(mask, q_shape)

		# --------------------------------------------------------------------------
		# Run model
		# --------------------------------------------------------------------------
		
		if mask is not None:
			q  = q  * mask
			db = db * tf.expand_dims(mask, 1)

		# Ensure masking didn't screw up the shape
		db = dynamic_assert_shape(db, db_shape)

		scores = tf.matmul(db, tf.expand_dims(q, 2))

		if use_dense:
			need_to_set_shape = scores.shape[1].value is None
			assert word_size is not None, "Cannot use_dense with unknown width_size"
			assert not need_to_set_shape or max_len is not None, f"Please supply max seq len since seq len {db.name} is dynamic"

			scores = tf.squeeze(scores, axis=2)

			if need_to_set_shape:
				delta = max_len - seq_len
				scores = tf.pad(scores, [[0, 0], [0, delta]])
				scores = tf.reshape(scores, [-1, max_len])
				
			scores = tf.layers.dense(scores, word_size)

			if need_to_set_shape:
				scores = scores[:,0:seq_len]

			scores = tf.expand_dims(scores, axis=2)
			scores = dynamic_assert_shape(scores, scores_shape)

		scores = tf.nn.softmax(scores, axis=1)
		scores = dynamic_assert_shape(scores, scores_shape)
		
		weighted_db = db * scores

		output = tf.reduce_sum(weighted_db, 1)
		output = dynamic_assert_shape(output, q_shape)

		if output_taps:
			return output, scores
		else:
			return output





