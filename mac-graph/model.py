
import tensorflow as tf
import numpy as np
import yaml

from .cell import execute_reasoning
from .encoder import encode_input
from .cell import *
from .util import *
from .hooks import *

def model_fn(features, labels, mode, params):

	# --------------------------------------------------------------------------
	# Setup input
	# --------------------------------------------------------------------------
	
	args = params
	
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

	# vocab_embedding = tf.Variable(tf.eye(args["vocab_size"], args["embed_width"]), name="vocab_embedding")

	tf.summary.image("vocab_embedding", tf.reshape(vocab_embedding, 
		[-1, args["vocab_size"], args["embed_width"], 1]))

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
		loss = tf.reduce_sum(crossent) / tf.to_float(features["d_batch_size"])
		
	# --------------------------------------------------------------------------
	# Optimize
	# --------------------------------------------------------------------------

	if mode == tf.estimator.ModeKeys.TRAIN:
		global_step = tf.train.get_global_step()
		optimizer = tf.train.AdamOptimizer(args["learning_rate"])
		train_op = minimize_clipped(optimizer, loss, args["max_gradient_norm"])


	# --------------------------------------------------------------------------
	# Predictions
	# --------------------------------------------------------------------------
	
	if mode in [tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL]:

		predicted_labels = tf.argmax(tf.nn.softmax(logits), axis=-1)

		predictions = {
			"predicted_label": predicted_labels,
			"actual_label": features["label"],
			"src":  features["src"],
			"type_string": features["type_string"],
		}

	# --------------------------------------------------------------------------
	# Eval metrics
	# --------------------------------------------------------------------------
	
	if mode == tf.estimator.ModeKeys.EVAL:

		eval_metric_ops = {
			"accuracy": tf.metrics.accuracy(labels=labels, predictions=predicted_labels),
		}

		with tf.gfile.GFile(args["types_path"]) as file:
			doc = yaml.load(file)
			print(doc)
			for type_string in doc:
				eval_metric_ops["z_accuracy_"+type_string] = tf.metrics.accuracy(
					labels=labels, 
					predictions=predicted_labels, 
					weights=tf.equal(features["type_string"], type_string))

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

