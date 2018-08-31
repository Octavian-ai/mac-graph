
import tensorflow as tf

from ..util import *
from ..attention import *
from ..input import UNK_ID
from ..minception import *
from ..args import ACTIVATION_FNS

# TODO: Make indicator row data be special token

def read_from_table(args, features, in_signal, noun, table, width, table_len=None, table_max_len=None):

	if args["read_indicator_cols"] > 0:
		ind_col = tf.get_variable(f"{noun}_indicator_col", [1, 1, args["read_indicator_cols"]])
		ind_col = tf.tile(ind_col, [features["d_batch_size"], tf.shape(table)[1], 1])
		table = tf.concat([table, ind_col], axis=2)
		width += args["read_indicator_cols"]

	# query = tf.layers.dense(in_signal, width, activation=tf.nn.tanh)
	query = tf.layers.dense(in_signal, width)

	output, score = attention(table, query,
		word_size=width, 
		table_len=table_len,
		table_max_len=table_max_len,
	)

	output = dynamic_assert_shape(output, [features["d_batch_size"], width])
	return output, score, table


def get_table_with_embedding(args, features, vocab_embedding, noun):
	
	# --------------------------------------------------------------------------
	# Constants and validations
	# --------------------------------------------------------------------------

	table = features[f"{noun}s"]
	table_len = features[f"{noun}s_len"]

	width = args[f"{noun}_width"]
	full_width = width * args["embed_width"]

	d_len = tf.shape(table)[1]
	assert table.shape[-1] == width


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
	
	emb_kb = tf.nn.embedding_lookup(vocab_embedding, table)
	emb_kb = dynamic_assert_shape(emb_kb, 
		[features["d_batch_size"], d_len, width, args["embed_width"]])

	emb_kb = tf.reshape(emb_kb, [-1, d_len, full_width])
	emb_kb = dynamic_assert_shape(emb_kb, 
		[features["d_batch_size"], d_len, full_width])

	return emb_kb, full_width, table_len


def read_from_table_with_embedding(args, features, vocab_embedding, in_signal, noun):
	"""Perform attention based read from table

	Will transform table into vocab embedding space
	
	@returns read_data
	"""

	with tf.name_scope(f"read_from_{noun}"):

		table, full_width, table_len = get_table_with_embedding(args, features, vocab_embedding, noun)

		# --------------------------------------------------------------------------
		# Read
		# --------------------------------------------------------------------------

		return read_from_table(args, features, 
			in_signal, 
			noun,
			table, 
			width=full_width, 
			table_len=table_len, 
			table_max_len=args[f"{noun}_max_len"])


def read_cell(args, features, vocab_embedding, 
	in_memory_state, in_control_state, in_data_stack, in_question_tokens, in_question_state):
	"""
	A read cell

	@returns read_data

	"""


	with tf.name_scope("read_cell"):

		# --------------------------------------------------------------------------
		# Read data
		# --------------------------------------------------------------------------

		in_signal = []

		# Commented out because it was hampering progress
		# if in_memory_state is not None and args["use_memory_cell"]:
		# 	in_signal.append(in_memory_state)

		# We may run the network with no control cell
		if in_control_state is not None and args["use_control_cell"]:
			in_signal.append(in_control_state)

		if args["read_from_question"]:
			in_signal.append(in_question_tokens[:,2])
			in_signal.append(in_question_tokens[:,6])

		if args["use_read_question_state"] or len(in_signal)==0:
			in_signal.append(in_question_state)

		in_signal = tf.concat(in_signal, -1)

		
		tap_attns = []
		tap_table = None

		taps = {}
		reads = []

		for j in range(args["read_heads"]):
			for i in ["kb_node", "kb_edge"]:
				if args[f"use_{i}"]:
					read, taps[i+"_attn"], table = read_from_table_with_embedding(
						args, 
						features, 
						vocab_embedding, 
						in_signal, 
						noun=i
					)

					read_words = tf.reshape(read, [features["d_batch_size"], args[i+"_width"], args["embed_width"]])
					
					if args["use_read_extract"]:
						d, taps[i+"_word_attn"] = attention_by_index(in_signal, read_words)
						d = tf.concat([d, in_signal], -1)
						d = tf.layers.dense(d, args["read_width"], activation=ACTIVATION_FNS[args["read_activation"]])
						reads.append(d)
					else:
						reads.append(read_words)
					

		if args["use_read_extract"]:
			reads = tf.stack(reads, axis=1)
			reads, taps["read_head_attn"] = attention_by_index(in_question_state, reads)
			# reads = tf.concat(reads, -1)
		else:
			reads = tf.concat(reads, -2)
			reads = tf.reshape(reads, [features["d_batch_size"], reads.shape[-1]*reads.shape[-2]])

		# --------------------------------------------------------------------------
		# Prepare and shape results
		# --------------------------------------------------------------------------
		
		# Residual skip connection
		out_data = tf.concat([reads, in_signal], -1)
		
		for i in range(args["read_layers"]):
			out_data = tf.layers.dense(out_data, args["read_width"])
			out_data = ACTIVATION_FNS[args["read_activation"]](out_data)
			
			if args["read_dropout"] > 0:
				out_data = tf.nn.dropout(out_data, 1.0-args["read_dropout"])


		return out_data, taps




