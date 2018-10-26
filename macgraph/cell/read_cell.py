
import tensorflow as tf

from ..util import *
from ..attention import *
from ..input import UNK_ID, get_table_with_embedding
from ..minception import *
from ..args import ACTIVATION_FNS

read_control_parts = ["token_content", "token_index", "step_const", "memory", "prev_output"]


# TODO: Make indicator row data be special token

def read_from_table(args, features, in_signal, noun, table, width, keys_len=None):

	if args["read_indicator_cols"] > 0:
		ind_col = tf.get_variable(f"{noun}_indicator_col", [1, 1, args["read_indicator_cols"]])
		ind_col = tf.tile(ind_col, [features["d_batch_size"], tf.shape(table)[1], 1])
		table = tf.concat([table, ind_col], axis=2)
		width += args["read_indicator_cols"]

	query = tf.layers.dense(in_signal, width)

	output, total_raw_score, taps = attention(table, query,
		key_width=width, 
		keys_len=keys_len,
		name="read_from_"+noun,
	)

	output = dynamic_assert_shape(output, [features["d_batch_size"], width])
	return output, table, total_raw_score, taps


def read_from_table_with_embedding(args, features, vocab_embedding, in_signal, noun):
	"""Perform attention based read from table

	Will transform table into vocab embedding space
	
	@returns read_data
	"""

	with tf.name_scope(f"read_from_{noun}"):

		table, full_width, keys_len = get_table_with_embedding(args, features, vocab_embedding, noun)

		# --------------------------------------------------------------------------
		# Read
		# --------------------------------------------------------------------------

		return read_from_table(args, features, 
			in_signal, 
			noun,
			table, 
			width=full_width, 
			keys_len=keys_len)


def read_cell(args, features, vocab_embedding, 
	in_memory_state, in_control_state, in_prev_outputs,
	in_question_tokens, in_question_state, 
	in_iter_id):
	"""
	A read cell

	@returns read_data

	"""

	attention_master_signal = tf.concat([in_iter_id, in_question_state], -1)
	
	def read_cell_query(name):
		with tf.name_scope(name):
			taps = {}
			sources = []

			def add_taps(prefix, extra_taps):
				for k, v in extra_taps.items():
					taps[prefix + "_" + k] = v

			token_query = tf.layers.dense(attention_master_signal, args["input_width"])
			token_signal, _, x_taps = attention(in_question_tokens, token_query)
			sources.append(token_signal)
			add_taps("token_content", x_taps)

			padding = [[0,0], [0, args["max_seq_len"] - tf.shape(in_question_tokens)[1]], [0,0]] # batch, seq_len, token
			in_question_tokens_padded = tf.pad(in_question_tokens, padding)
			in_question_tokens_padded.set_shape([None, args["max_seq_len"], None])

			token_index_signal, query = attention_by_index(in_question_tokens_padded, attention_master_signal)
			sources.append(token_index_signal)
			taps["token_index_attn"] = tf.expand_dims(query, 2)

			step_const_signal = tf.layers.dense(in_iter_id, args["input_width"])
			sources.append(step_const_signal)
			
			if args["use_memory_cell"]:
				memory_shape = [features["d_batch_size"], args["memory_width"] // args["input_width"], args["input_width"]]
				memory_query = tf.layers.dense(attention_master_signal, args["input_width"])
				memory_signal, _, x_taps  = attention(tf.reshape(in_memory_state, memory_shape), memory_query)

				sources.append(memory_signal)
				add_taps("memory", x_taps)

			prev_output_query = tf.layers.dense(attention_master_signal, args["output_classes"])
			in_prev_outputs_padded = tf.pad(in_prev_outputs, [[0,0],[0, args["max_decode_iterations"] - tf.shape(in_prev_outputs)[1]],[0,0]])
			prev_output_signal, _, x_taps = attention(in_prev_outputs_padded, prev_output_query)
			sources.append(prev_output_signal)
			add_taps("prev_output", x_taps)

			query_signal, q_tap = attention_by_index(tf.stack(sources, 1), attention_master_signal)
			taps["switch_attn"] = q_tap

			return query_signal, taps


	with tf.name_scope("read_cell"):

		tap_attns = []
		tap_table = None
		taps = {}

		reads = []
		attn_focus = []

		head_total = args["read_heads"] * len(args["kb_list"])

		if head_total == 0:
			return tf.fill([features["d_batch_size"], 1], 0.0), {}

		# --------------------------------------------------------------------------
		# Read data
		# --------------------------------------------------------------------------

		# in_signal = []

		# # Commented out because it was hampering progress
		# if in_memory_state is not None and args["use_memory_cell"]:
		# 	in_signal.append(in_memory_state)

		# # We may run the network with no control cell
		# if in_control_state is not None and args["use_control_cell"]:
		# 	if args["use_read_control_share"]:
		# 		in_signal.append(in_control_state)
		# 	else:
		# 		control_head_count = args["control_width"] // args["input_width"]
		# 		control_heads_per_read_head = control_head_count // head_total
		# 		assert args["control_width"] % args["input_width"] == 0, "If not sharing control heads between read heads, the control width must be integer multiple of input_width"
		# 		assert control_head_count % head_total == 0, f"If not sharing control heads {control_head_count} then number of control heads must be multiple of read heads {head_total}"
		# 		control_heads = tf.reshape(in_control_state, [features["d_batch_size"], head_total, control_heads_per_read_head * args["input_width"]])

		# if args["use_read_question_state"] or len(in_signal)==0:
		# 	in_signal.append(in_question_state)

		# head_i = 0

		for j in range(args["read_heads"]):
			for i in args["kb_list"]:

				# if args["use_read_control_share"]:
				# 	in_signal_to_head = tf.concat(in_signal, -1)
				# else:
				# 	control_head_slice = control_heads[:,head_i,:]
				# 	in_signal_to_head = tf.concat(in_signal + [control_head_slice], -1)

				read_query, rcq_taps = read_cell_query(i + str(j))

				read, table, score_raw_total, read_table_taps = read_from_table_with_embedding(
					args, 
					features, 
					vocab_embedding, 
					read_query, 
					noun=i
				)

				for k,v in {**read_table_taps, **rcq_taps}.items():
					taps[i + str(j) + "_" + k] = v

				attn_focus.append(score_raw_total)

				read_words = tf.reshape(read, [features["d_batch_size"], args[i+"_width"], args["embed_width"]])	
			
				d, taps[i + str(j) + "_word_attn"] = attention_by_index(read_words, attention_master_signal, name=i+"_word_attn")
				d = tf.concat([d, attention_master_signal], -1)
				d = tf.layers.dense(d, args["read_width"], activation=ACTIVATION_FNS[args["read_activation"]])
				reads.append(d)
				

				# head_i += 1
	
		reads = tf.stack(reads, axis=1)
		read_word, taps["read_head_attn"] = attention_by_index(reads, attention_master_signal, name="read_head_attn")
	
		# --------------------------------------------------------------------------
		# Prepare and shape results
		# --------------------------------------------------------------------------
		
		taps["read_head_attn_focus"] = tf.concat(attn_focus, -1)

		# Residual skip connection
		out_data = tf.concat([read_word, attention_master_signal] + attn_focus, -1)
		
		for i in range(args["read_layers"]):
			out_data = tf.layers.dense(out_data, args["read_width"])
			out_data = ACTIVATION_FNS[args["read_activation"]](out_data)
			
			if args["read_dropout"] > 0:
				out_data = tf.nn.dropout(out_data, 1.0-args["read_dropout"])


		return out_data, taps




