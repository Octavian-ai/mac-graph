
import tensorflow as tf

from ..util import *

from ..attention import *


def read_from_graph(args, features, vocab_embedding, query, mask=None, name="read_from_graph"):
	"""Perform attention based read from table

	@param W_score is for testing/debug purposes so you can easily inject the score fn you'd like. The code will default to a variable normally
	
	@returns read_data
	"""

	with tf.name_scope(name):

		# --------------------------------------------------------------------------
		# Constants and validations
		# --------------------------------------------------------------------------

		kb_full_width = args["kb_width"] * args["embed_width"]
		d_kb_len = tf.shape(features["knowledge_base"])[1]

		assert_shape(query, [kb_full_width])
		assert features["knowledge_base"].shape[-1] == args["kb_width"]

		if mask is not None:
			assert_shape(mask,  [kb_full_width])
			assert mask.dtype == tf.float32, "Mask should be floats between 0 and 1"


		# --------------------------------------------------------------------------
		# Embed graph tokens
		# --------------------------------------------------------------------------
		
		emb_kb = tf.nn.embedding_lookup(vocab_embedding, features["knowledge_base"])
		emb_kb = dynamic_assert_shape(emb_kb, 
			[features["d_batch_size"], d_kb_len, args["kb_width"], args["embed_width"]])

		emb_kb = tf.reshape(emb_kb, [-1, d_kb_len, kb_full_width])

		tf.summary.image("db", tf.expand_dims(emb_kb, -1))

		# --------------------------------------------------------------------------
		# Do lookup via attention
		# --------------------------------------------------------------------------

		output = attention(args, query, emb_kb, mask)
		return output



def read_cell(args, features, in_memory_state, in_control, vocab_embedding):
	"""
	A read cell

	@returns read_data

	"""

	assert_shape(in_memory_state, [args["bus_width"]])
	assert_shape(in_control,      [args["bus_width"]])

	in_all = tf.concat([in_memory_state, in_control], -1)
	w = args["embed_width"] * args["kb_width"]

	query = tf.layers.dense(in_all, w, activation=tf.nn.tanh)
	# query = tf.layers.dense(query,  w, activation=tf.nn.tanh)

	mask  = tf.layers.dense(in_all, w, activation=tf.nn.tanh)
	# mask  = tf.layers.dense(mask,   w, activation=tf.nn.tanh)

	read_data = read_from_graph(args, features, vocab_embedding, query, mask)

	# --------------------------------------------------------------------------
	# Shrink results
	# --------------------------------------------------------------------------

	read_data = tf.layers.dense(read_data, args["bus_width"], name="data_read_shrink")
	read_data = dynamic_assert_shape(read_data, [features["d_batch_size"], args["bus_width"]])


	return read_data




