

import tensorflow as tf

def basic_cell(args, i, unit_mul):

	c = tf.contrib.rnn.LSTMCell(int(args['bus_width']*unit_mul))
	c = tf.contrib.rnn.DropoutWrapper(c, args['dropout'])

	if i > 1:
		c = tf.contrib.rnn.ResidualWrapper(c)

	return c

def cell_stack(args, layer_mul=1, unit_mul=1):
	cells = []
	for i in range(int(args["num_input_layers"]*layer_mul)):
		cells.append(basic_cell(args, i, unit_mul))

	cell = tf.contrib.rnn.MultiRNNCell(cells)
	return cell


def encode_input(args, features):
	"""

	Expects data in time-minor format (e.g. batch-major)

	Arguments:
		- features {
			"src":     Tensor([batch_size, seq_len]), 
			"src_len": Tensor([batch_size])
		}

	Returns:
		(
			Tensor("outputs",     [batch_size, seq_len, bus_width * 2]), 
			Tensor("final_state", [batch_size, bus_width * 2])
		)
	"""

	# --------------------------------------------------------------------------
	# Setup inputs
	# --------------------------------------------------------------------------

	assert "src" in features
	assert "src_len" in features

	dynamic_batch_size = tf.shape(features["src"])[0]
	time_major = False

	# Trim down to the residual batch size (e.g. when at end of input data)
	padded_src_len = features["src_len"][0 : dynamic_batch_size]


	# --------------------------------------------------------------------------
	# Embed vocab
	# --------------------------------------------------------------------------

	vocab_embedding = tf.get_variable("vocab_embedding", [args["vocab_size"], args["bus_width"]], tf.float32)
	src  = tf.nn.embedding_lookup(vocab_embedding, tf.transpose(features["src"]))

	# --------------------------------------------------------------------------
	# Encoder
	# --------------------------------------------------------------------------
	
	# 1/2 multiplier so that when we concat the layers together we get bus_width
	fw_cell = cell_stack(args, unit_mul=0.5)
	bw_cell = cell_stack(args, unit_mul=0.5)
	
	(fw_output, bw_output), (fw_states, bw_states) = tf.nn.bidirectional_dynamic_rnn(
		fw_cell,
		bw_cell,
		src,
		dtype=tf.float32,
		sequence_length=padded_src_len,
		time_major=time_major,
		swap_memory=True)

	encoder_outputs = tf.concat( (fw_output, bw_output), axis=-1)
	encoder_final_state = tf.concat( (fw_states, bw_states), axis=-1)
	
	return (encoder_outputs, encoder_final_state)





