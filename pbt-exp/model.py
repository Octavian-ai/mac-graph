import tensorflow as tf
import numpy as np

import traceback
from functools import reduce

from pbt import gen_scaffold
from dnc import DNC

def score_to_class(tensor, buckets=2):
	return tf.cast(tf.round(tensor * (buckets-1)), tf.int32)

def model_fn(features, labels, mode, params):

	# --------------------------------------------------------------------------
	# Model
	# --------------------------------------------------------------------------

	access_config = {
		"memory_size": 16,
		"word_size": 16,
		"num_reads": 4,
		"num_writes": 1,
	}
	
	controller_config = {
			"hidden_size": 64,
	}

	clip_value = 20

	dnc_core = DNC(access_config, controller_config, 5, clip_value)
	initial_state = dnc_core.initial_state(params["batch_size"])
	output_logits, _ = tf.nn.dynamic_rnn(
			cell=dnc_core,
			inputs=features,
			time_major=True,
			initial_state=initial_state)

	# --------------------------------------------------------------------------
	# Build EstimatorSpec
	# --------------------------------------------------------------------------

	train_loss = params["dataset_"+mode].cost(output_logits, labels["target"], labels["mask"])

	# Set up optimizer with global norm clipping.
	trainable_variables = tf.trainable_variables()
	grads, _ = tf.clip_by_global_norm(
			tf.gradients(train_loss, trainable_variables), params["max_grad_norm"])

	global_step = tf.get_variable(
			name="global_step",
			shape=[],
			dtype=tf.int64,
			initializer=tf.zeros_initializer(),
			trainable=False,
			collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

	optimizer = tf.train.RMSPropOptimizer(
			params["lr"], epsilon=params["optimizer_epsilon"])
	
	train_step = optimizer.apply_gradients(
			zip(grads, trainable_variables), global_step=global_step)

	
	# dataset_tensors_np, output_np = sess.run([dataset_tensors, output])
	# dataset_string = dataset.to_human_readable(dataset_tensors_np, output_np)

	output_sigmoid = tf.nn.sigmoid(output_logits)
	delta = tf.abs(output_sigmoid - labels["target"])
	tf.summary.histogram("delta", delta)
	equality = tf.cast(delta < 0.1, tf.float32) * tf.expand_dims(labels["mask"],-1)

	correct_elements = tf.reduce_mean(tf.reduce_sum(equality, [0,2]))
	pct_correct = tf.reduce_mean(tf.reduce_sum(equality, [0,2]) / tf.cast(labels["total_targ_batch"], tf.float32))

	eval_metric_ops = {
		"accuracy": tf.metrics.mean(pct_correct),
		"loss": tf.metrics.mean(train_loss),
		"correct_elements": tf.metrics.mean(correct_elements),
		"total_elements": tf.metrics.mean(tf.cast(labels["total_targ_batch"], tf.float32))
	}

	image_mask = tf.expand_dims(tf.expand_dims(labels["mask"],-1),-1)
	
	xent = tf.expand_dims(
		tf.nn.sigmoid_cross_entropy_with_logits(labels=labels["target"], logits=output_logits * tf.expand_dims(labels["mask"],-1)), -1)
	
	image = tf.concat([
		# tf.expand_dims(output_logits, -1),
		output_sigmoid, 
		labels["target"],
		# tf.expand_dims(equality, -1),
		# xent / tf.reduce_max(xent)
	], -1)
	# tf summary image expects shape [batch_size, height, width, channels]
	image = tf.transpose(image, perm=[1,0,2])
	tf.summary.image("output_compare", tf.expand_dims(image, -1), 4)
	
	tf.summary.scalar("train_loss", tf.reduce_mean(train_loss))
	tf.summary.scalar("train_accuracy", pct_correct)
	tf.summary.scalar("correct_elements", correct_elements)
	tf.summary.scalar("total_elements", tf.reduce_mean(labels["total_targ_batch"], axis=-1))

	tf.summary.scalar("max_length", tf.convert_to_tensor(params["dataset_"+mode]._max_length))
	tf.summary.scalar("max_repeats", tf.convert_to_tensor(params["dataset_"+mode]._max_repeats))

	return tf.estimator.EstimatorSpec(
		mode, 
		loss=train_loss, 
		train_op=train_step, 
		eval_metric_ops=eval_metric_ops,
		scaffold=gen_scaffold(params)
	)

	# --------------------------------------------------------------------------


	


