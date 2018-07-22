
from .util import *

def attention(database, query, mask=None, word_size=None, use_dense=False, output_taps=False):
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

		if use_dense:
			assert q.shape[-1] is not None, "Cannot use_dense with unknown width query"
			q = tf.layers.dense(q, word_size)

		scores = tf.matmul(db, tf.expand_dims(q, 2))
		scores = tf.nn.softmax(scores, axis=1)
		scores = dynamic_assert_shape(scores, (batch_size, seq_len, 1))
		
		# barcode_height = tf.cast(tf.round(tf.div(tf.cast(seq_len, tf.float32), 3.0)), tf.int32)
		# barcode_image = tf.tile(tf.reshape(scores, [batch_size, 1, seq_len, 1]), [1, barcode_height, 1, 1])

		# tf.summary.image("attention", 
		# 	vector_to_barcode(scores),# barcode_image,
		# 	max_outputs=1, 
		# 	family="Attention")

		weighted_db = db * scores

		output = tf.reduce_sum(weighted_db, 1)
		output = dynamic_assert_shape(output, q_shape)

		if output_taps:
			return output, scores
		else:
			return output





