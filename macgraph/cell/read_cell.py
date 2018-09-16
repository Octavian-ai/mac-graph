
import tensorflow as tf

from ..util import *
from ..attention import *
from ..input import UNK_ID
from ..minception import *
from ..args import ACTIVATION_FNS

# TODO: Make indicator row data be special token

def read_from_table(args, features, in_signal, noun, table, width, table_len=None):

	if args["read_indicator_cols"] > 0:
		ind_col = tf.get_variable(f"{noun}_indicator_col", [1, 1, args["read_indicator_cols"]])
		ind_col = tf.tile(ind_col, [features["d_batch_size"], tf.shape(table)[1], 1])
		table = tf.concat([table, ind_col], axis=2)
		width += args["read_indicator_cols"]

	if args["use_read_query_block"]:
		# Perform word copy op at the block level, should be easier to train
		assert in_signal.shape[-1] is not None, "input signal width must be known"
		assert in_signal.shape[-1] % args["embed_width"] == 0, "in_signal size must be mulitple of embed_width"
		assert width % args["embed_width"] == 0, f"table width {width} must be multiple of embed_width {args['embed_width']}"

		n_input_blocks = in_signal.shape[-1] // args["embed_width"]
		in_signal_blocked = tf.reshape(in_signal, [features["d_batch_size"], n_input_blocks, args["embed_width"]])
		n_query_blocks = width // args["embed_width"]

		query_proj_w = tf.get_variable(noun+"_query_proj_w", [n_query_blocks, n_input_blocks])
		query_proj_b = tf.get_variable(noun+"_query_proj_b", [n_query_blocks, args["embed_width"]])
		query = tf.einsum('bie,qi->bqe', in_signal_blocked, query_proj_w)
		# query = tf.matmul(in_signal_blocked, query_proj_w)
		query += query_proj_b
		query = tf.reshape(query, [features["d_batch_size"], n_query_blocks * args["embed_width"]])
	else:
		query = tf.layers.dense(in_signal, width)

	output, score_sm, total_raw_score = attention(table, query,
		word_size=width, 
		table_len=table_len,
	)

	output = dynamic_assert_shape(output, [features["d_batch_size"], width])
	return output, score_sm, table, total_raw_score


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
			table_len=table_len)


def read_cell(args, features, vocab_embedding, 
	in_memory_state, in_control_state, in_data_stack, in_question_tokens, in_question_state):
	"""
	A read cell

	@returns read_data

	"""


	with tf.name_scope("read_cell"):

		tap_attns = []
		tap_table = None

		taps = {}
		reads = []
		attn_focus = []

		head_total = args["read_heads"] * len(args["kb_list"])

		# --------------------------------------------------------------------------
		# Read data
		# --------------------------------------------------------------------------

		in_signal = []

		# Commented out because it was hampering progress
		if in_memory_state is not None and args["use_memory_cell"]:
			in_signal.append(in_memory_state)

		# We may run the network with no control cell
		if in_control_state is not None and args["use_control_cell"]:
			if args["use_read_control_share"]:
				in_signal.append(in_control_state)
			else:
				control_head_count = args["control_width"] // args["input_width"]
				control_heads_per_read_head = control_head_count // head_total
				assert args["control_width"] % args["input_width"] == 0, "If not sharing control heads between read heads, the control width must be integer multiple of input_width"
				assert control_head_count % head_total == 0, "If not sharing control heads then number of control heads must be multiple of read heads"
				control_heads = tf.reshape(in_control_state, [features["d_batch_size"], head_total, control_heads_per_read_head * args["input_width"]])

		if args["use_read_question_state"] or len(in_signal)==0:
			in_signal.append(in_question_state)



		head_i = 0
		for j in range(args["read_heads"]):
			for i in args["kb_list"]:

				if args["use_read_control_share"]:
					in_signal_to_head = tf.concat(in_signal, -1)
				else:
					control_head_slice = control_heads[:,head_i,:]
					in_signal_to_head = tf.concat(in_signal + [control_head_slice], -1)

				read, taps[i+"_attn"], table, score_raw_total = read_from_table_with_embedding(
					args, 
					features, 
					vocab_embedding, 
					in_signal_to_head, 
					noun=i
				)

				attn_focus.append(score_raw_total)

				read_words = tf.reshape(read, [features["d_batch_size"], args[i+"_width"], args["embed_width"]])
				
				if args["use_read_extract"]:
					d, taps[i+"_word_attn"] = attention_by_index(in_signal_to_head, read_words)
					d = tf.concat([d, in_signal_to_head], -1)
					d = tf.layers.dense(d, args["read_width"], activation=ACTIVATION_FNS[args["read_activation"]])
					reads.append(d)
				else:
					reads.append(read_words)

				head_i += 1
					

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
		
		taps["read_head_attn_focus"] = tf.concat(attn_focus, -1)

		# Residual skip connection
		out_data = tf.concat([reads] + in_signal + attn_focus, -1)
		
		for i in range(args["read_layers"]):
			out_data = tf.layers.dense(out_data, args["read_width"])
			out_data = ACTIVATION_FNS[args["read_activation"]](out_data)
			
			if args["read_dropout"] > 0:
				out_data = tf.nn.dropout(out_data, 1.0-args["read_dropout"])


		return out_data, taps




