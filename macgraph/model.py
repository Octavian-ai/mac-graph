
import tensorflow as tf
import numpy as np
import yaml

from .cell import execute_reasoning
from .encoder import encode_input
from .cell import *
from .util import *
from .hooks import *
from .input import *
from .optimizer import *

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

	vocab = Vocab.load(args)

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


	logits = execute_reasoning(args, 
		features=features, 
		question_state=question_state,
		labels=labels,
		question_tokens=question_tokens, 
		vocab_embedding=vocab_embedding)

	tf.summary.image("logits", tf.expand_dims(tf.expand_dims(tf.nn.softmax(logits), 1), -1))

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

		learning_rate = args["learning_rate"]

		if args["use_lr_finder"]:
			learning_rate = tf.train.exponential_decay(
				1E-06, 
				global_step,
				decay_steps=1000, 
				decay_rate=1.1)

		elif args["use_lr_decay"]:
			learning_rate = args["learning_rate"] - tf.train.exponential_decay(
				args["learning_rate"], 
				global_step,
				decay_steps=10000, 
				decay_rate=0.9)



		tf.summary.scalar("learning_rate", learning_rate, family="hyperparam")
		tf.summary.scalar("current_step", global_step, family="hyperparam")

		var = tf.trainable_variables()
		gradients = tf.gradients(loss, var)
		norms = [tf.norm(i, 2) for i in gradients if i is not None]

		tf.summary.histogram("grad_norm", norms)
		tf.summary.scalar("grad_norm", tf.reduce_max(norms), family="hyperparam")

		optimizer = tf.train.AdamOptimizer(learning_rate)
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

	
		try:
			with tf.gfile.GFile(args["question_types_path"]) as file:
				doc = yaml.load(file)
				for type_string in doc.keys():
					if args["type_string_prefix"] is None or type_string.startswith(args["type_string_prefix"]):
						eval_metric_ops["type_accuracy_"+type_string] = tf.metrics.accuracy(
							labels=labels, 
							predictions=predicted_labels, 
							weights=tf.equal(features["type_string"], type_string))


			print("output_classes")
			with tf.gfile.GFile(args["output_classes_path"]) as file:
				doc = yaml.load(file)
				print("output_classes", doc)
				for answer_class in doc.keys():
					e = vocab.lookup(pretokenize_json(answer_class))
					weights = tf.equal(labels, tf.cast(e, tf.int64))
					eval_metric_ops["class_accuracy_"+str(answer_class)] = tf.metrics.accuracy(
						labels=labels, 
						predictions=predicted_labels, 
						weights=weights)

		except tf.errors.NotFoundError as err:
			print(err)
			pass
		except Exception as err:
			print(err)
			pass


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
