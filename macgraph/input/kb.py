
import tensorflow as tf

from ..util import dynamic_assert_shape
from .text_util import UNK_ID


def get_table_with_embedding(args, features, vocab_embedding, noun):
	
	# --------------------------------------------------------------------------
	# Constants and validations
	# --------------------------------------------------------------------------

	# TODO: remove these pesky ses
	table = features[f"{noun}s"]
	table_len = features[f"{noun}s_len"]
	width = args[f"{noun}_width"]
	full_width = width * args["embed_width"]

	d_len = tf.shape(table)[1]
	assert table.shape[-1] == width, f"Table shape {table.shape} did not have expected inner width dimensions of {width}"


	# --------------------------------------------------------------------------
	# Extend table if desired
	# --------------------------------------------------------------------------

	if args["read_indicator_rows"] > 0:
		# Add a trainable row to the table
		ind_row_shape = [features["d_batch_size"], args["read_indicator_rows"], width]
		ind_row = tf.fill(ind_row_shape, tf.cast(UNK_ID, table.dtype))
		table = tf.concat([table, ind_row], axis=1)
		table_len += args["read_indicator_rows"]
		d_len += args["read_indicator_rows"]

	# --------------------------------------------------------------------------
	# Embed graph tokens
	# --------------------------------------------------------------------------
	
	table = dynamic_assert_shape(table, [features["d_batch_size"], d_len, width])

	emb_kb = tf.nn.embedding_lookup(vocab_embedding, table)
	emb_kb = dynamic_assert_shape(emb_kb, 
		[features["d_batch_size"], d_len, width, args["embed_width"]])

	emb_kb = tf.reshape(emb_kb, [-1, d_len, full_width])
	emb_kb = dynamic_assert_shape(emb_kb, 
		[features["d_batch_size"], d_len, full_width])

	return emb_kb, full_width, table_len

	