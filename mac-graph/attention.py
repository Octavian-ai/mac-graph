
from .util import *

def attention(args, query, database, mask=None, use_dense=True):
	"""
	Apply attention

	Arguments:
		- `query` shape (batch_size, width)
		- `database` shape (batch_size, len, width)
		- Optional `mask` shape (batch_size, width)
		- use_dense Whether to instantiate a free variable for comparison function

	"""

	q = query
	db = database

	# --------------------------------------------------------------------------
	# Validate inputs
	# --------------------------------------------------------------------------

	assert len(database.shape) == 3, "Database should be shape [batch, len, width]"

	batch_size = tf.shape(db)[0]
	seq_len = tf.shape(db)[1]
	word_size = tf.shape(db)[2]

	q = dynamic_assert_shape(q, (batch_size, word_size) )

	# --------------------------------------------------------------------------
	# Run model
	# --------------------------------------------------------------------------
	
	if mask is not None:
		q  = q  * mask
		db = db * tf.expand_dims(mask, 1)


	db = dynamic_assert_shape(db, tf.shape(database))

	embed_width = args["embed_width"]

	tf.summary.image("masked_q", tf.reshape(q, [batch_size, -1, embed_width ,1]))
	tf.summary.image("masked_db", tf.reshape(db, [batch_size, seq_len, word_size ,1]))

	if use_dense:
		assert q.shape[-1] is not None, "Cannot use_dense with unknown width query"
		q = tf.layers.dense(q, q.shape[-1])

	scores = tf.matmul(db, tf.expand_dims(q, 2))
	scores = tf.nn.softmax(scores, axis=1)
	scores = dynamic_assert_shape(scores, (batch_size, seq_len, 1))
	tf.summary.image("attn_scores_sm", tf.reshape(scores, [batch_size, 1, seq_len, 1]))

	weighted_db = db * scores

	output = tf.reduce_sum(weighted_db, 1)
	output = dynamic_assert_shape(output, (batch_size, word_size))

	return output





