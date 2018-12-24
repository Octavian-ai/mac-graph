
import tensorflow as tf

from ..attention import *
from .types import *

def generate_token_index_query(context:CellContext, name:str):
	with tf.name_scope(name):
		with tf.variable_scope(name):

			taps = {}

			master_signal = context.in_iter_id

			padding = [[0,0], [0, tf.maximum(0,context.args["max_seq_len"] - tf.shape(context.in_question_tokens)[1])], [0,0]] # batch, seq_len, token
			in_question_tokens_padded = tf.pad(context.in_question_tokens, padding)
			in_question_tokens_padded.set_shape([None, context.args["max_seq_len"], None])

			token_index_signal, query = attention_by_index(in_question_tokens_padded, None)
			
			output = token_index_signal
			taps["token_index_attn"] = tf.expand_dims(query, 2)

			return output, taps



def generate_query(context:CellContext, name):
	with tf.name_scope(name):

		taps = {}
		sources = []

		def add_taps(prefix, extra_taps):
			for k, v in extra_taps.items():
				taps[prefix + "_" + k] = v

		# --------------------------------------------------------------------------
		# Produce all the difference sources of addressing query
		# --------------------------------------------------------------------------

		ms = [context.in_iter_id]

		if context.args["use_memory_cell"]:
			ms.append(context.in_memory_state)

		if context.args["use_question_state"]:
			ms.append(context.in_question_state)

		master_signal = tf.concat(ms, -1)

		# Content address the question tokens
		token_query = tf.layers.dense(master_signal, context.args["input_width"])
		token_signal, _, x_taps = attention(context.in_question_tokens, token_query)
		sources.append(token_signal)
		add_taps("token_content", x_taps)

		# Index address the question tokens
		padding = [[0,0], [0, tf.maximum(0,context.args["max_seq_len"] - tf.shape(context.in_question_tokens)[1])], [0,0]] # batch, seq_len, token
		in_question_tokens_padded = tf.pad(context.in_question_tokens, padding)
		in_question_tokens_padded.set_shape([None, context.args["max_seq_len"], None])

		token_index_signal, query = attention_by_index(in_question_tokens_padded, master_signal)
		sources.append(token_index_signal)
		taps["token_index_attn"] = tf.expand_dims(query, 2)

		# Use the iteration id
		# Disabling for now as not useful
		# step_const_signal = tf.layers.dense(context.in_iter_id, context.args["input_width"])
		# taps["step_const_signal"] = step_const_signal
		# sources.append(step_const_signal)
		
		# Use the memory contents
		# For now disabled as it forces memory to be too big and is not currently used
		# if context.args["use_memory_cell"]:
		# 	memory_shape = [context.features["d_batch_size"], context.args["memory_width"] // context.args["input_width"], context.args["input_width"]]
		# 	memory_query = tf.layers.dense(master_signal, context.args["input_width"])
		# 	memory_signal, _, x_taps  = attention(tf.reshape(context.in_memory_state, memory_shape), memory_query)

		# 	sources.append(memory_signal)
		# 	add_taps("memory", x_taps)

		if context.args["use_read_previous_outputs"]:
			# Use the previous output of the network
			prev_output_query = tf.layers.dense(master_signal, context.args["output_width"])
			in_prev_outputs_padded = tf.pad(context.in_prev_outputs, [[0,0],[0, context.args["max_decode_iterations"] - tf.shape(context.in_prev_outputs)[1]],[0,0]])
			prev_output_signal, _, x_taps = attention(in_prev_outputs_padded, prev_output_query)
			sources.append(prev_output_signal)
			add_taps("prev_output", x_taps)

		# --------------------------------------------------------------------------
		# Choose a query source
		# --------------------------------------------------------------------------

		query_signal, q_tap = attention_by_index(tf.stack(sources, 1), master_signal)
		taps["switch_attn"] = q_tap

		return query_signal, taps
