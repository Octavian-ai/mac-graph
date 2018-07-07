
import tensorflow as tf
import numpy as np

class FloydHubMetricHook(tf.train.SessionRunHook):
	"""An easy way to output your metric_ops to FloydHub's training metric graphs

	This is designed to fit into TensorFlow's EstimatorSpec. Assuming you've
	already defined some metric_ops for monitoring your training/evaluation,
	this helper class will compute those operations then print them out in 
	the format that FloydHub is expecting. For example:

	```
	def model_fn(features, labels, mode, params):

		# Set up your model
		loss = ...
		my_predictions = ...

		eval_metric_ops = {
			"accuracy": tf.metrics.accuracy(labels=labels, predictions=my_predictions)
			"loss": tf.metrics.mean(loss)
		}

		return EstimatorSpec(mode,
			eval_metric_ops = eval_metric_ops,

			# **Here it is! The magic!! **
			eval_hooks = [FloydHubMetricHook(eval_metric_ops)]

		)
	```

	FloydHubMetricHook has one optional parameter, *prefix* for using it multiple times
	(e.g. prefix="train_" for training metrics, prefix="eval_" for evaluation metrics).
	

	"""

	def __init__(self, metric_ops, prefix=""):
		self.metric_ops = metric_ops
		self.prefix = prefix
		self.readings = {}

	def before_run(self, run_context):
		return tf.train.SessionRunArgs(self.metric_ops)

	def after_run(self, run_context, run_values):
		if run_values.results is not None:
			for k,v in run_values.results.items():
				try:
					self.readings[k].append(v[1])
				except KeyError:
					self.readings[k] = [v[1]]

	def end(self, session):
		for k, v in self.readings.items():
			a = np.average(v)
			print(f'{{"metric": "{self.prefix}{k}", "value": {a}}}')

		self.readings = {}
