
import tensorflow as tf
import numpy as np

from .cell import execute_reasoning
from .encoder import encode_input
from .util import *
from .hooks import *

def model_fn(features, labels, mode, params):

	# --------------------------------------------------------------------------
	# Setup input
	# --------------------------------------------------------------------------
	
	args = params
	dynamic_batch_size = tf.shape(features["src"])[0]

	# EstimatorSpec slots
	loss = None
	train_op = None
	eval_metric_ops = None
	predictions = None
	eval_hooks = None

	# --------------------------------------------------------------------------
	# Shared variables
	# --------------------------------------------------------------------------
	
	vocab_embedding = tf.get_variable(
		"vocab_embedding", 
		[args["vocab_size"], args["embed_width"]], 
		tf.float32)

	# --------------------------------------------------------------------------
	# Model for realz
	# --------------------------------------------------------------------------
	
	question_tokens, question_state = encode_input(args, features, vocab_embedding)

	logits = execute_reasoning(args, features, labels,
		question_tokens=question_tokens, 
		question_state=question_state,
		vocab_embedding=vocab_embedding)
	
	# --------------------------------------------------------------------------
	# Calc loss
	# --------------------------------------------------------------------------	

	if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:		
		crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

		loss = tf.reduce_sum(crossent) / tf.to_float(dynamic_batch_size)
		# loss = tf.reduce_sum(tf.cast(question_state, tf.float32) * tf.get_variable("W", dtype=tf.float32, shape=[1]))

	# --------------------------------------------------------------------------
	# Optimize
	# --------------------------------------------------------------------------

	if mode == tf.estimator.ModeKeys.TRAIN:
		global_step = tf.train.get_global_step()
		optimizer = tf.train.AdamOptimizer(args["learning_rate"])
		train_op = minimize_clipped(optimizer, loss, args["max_gradient_norm"])


	# --------------------------------------------------------------------------
	# Eval metrics
	# --------------------------------------------------------------------------
	
	if mode ==  tf.estimator.ModeKeys.EVAL:

		predictions = tf.argmax(logits, axis=-1)

		eval_metric_ops = {
			"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions),
			"loss_": tf.metrics.mean(loss), # For FloydHub
		}

		eval_hooks = [FloydHubMetricHook(eval_metric_ops)]





	return tf.estimator.EstimatorSpec(mode,
		loss=loss,
		train_op=train_op,
		predictions=predictions,
		eval_metric_ops=eval_metric_ops,
		export_outputs=None,
		training_chief_hooks=None,
		training_hooks=None,
		scaffold=None,
		evaluation_hooks=eval_hooks,
		prediction_hooks=None
	)

