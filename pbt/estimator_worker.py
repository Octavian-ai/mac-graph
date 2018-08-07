
import tensorflow as tf
import numpy as np

import traceback
import os.path

from .worker import Worker
from .param import *
from .params import *

"""### Train the model"""

def resize_and_load(var, val, sess):
	o_shape = var.get_shape().as_list()
	i_shape = list(val.shape)
				
	if o_shape != i_shape:
		resize_dim = 1 # may not always hold true, assumption for now
		delta = o_shape[resize_dim] - i_shape[resize_dim]
		
		if delta != 0:
			tf.logging.info("reshape var {} by {}".format(var.name, deta))

		if delta < 0:
			val = val[:,:o_shape[1]]
		elif delta > 0:
			val = np.pad(val, ((0,0),(0, delta)), 'reflect')

		v.load(val, self.sess)


def gen_scaffold(params):
	def init_fn(scaffold, session):
		tf.logging.info("Running Scaffold init_fn", params)

		vs = params["vars"]

		if vs is not None: 
			for var in tf.trainable_variables():
				if var.name in vs:

					val = vs[var.name]
					resize_and_load(var, val, session)

	# return tf.train.Scaffold(init_fn=lambda scaffold, session: True)
	return tf.train.Scaffold(init_fn=init_fn)



class MetricHook(tf.train.SessionRunHook):
	def __init__(self, metrics, cb, key=0):
		self.metrics = metrics
		self.key = key
		self.readings = []

	def before_run(self, run_context):
		return tf.train.SessionRunArgs(self.metrics)

	def after_run(self, run_context, run_values):
		if run_values.results is not None:
			self.readings.append(run_values.results[self.key][1])

	def end(self, session):
		if len(self.readings) > 0:
			self.cb(np.average(self.readings))
			self.readings.clear()



class EstimatorWorker(Worker):
	
	def __init__(self, init_params, hyperparam_spec):
		self.estimator = None
		self.trained = False
		self.use_warm_start = True

		if self.use_warm_start:
			assert "model_id" in hyperparam_spec, "Warm start requires model_id hyperparam"
		else:
			assert "vars" in hyperparam_spec, "Cold start requires vars hyperparam"

		super().__init__(init_params, hyperparam_spec)


	def setup_estimator(self):
	
		if self.use_warm_start:
			try:
				model_dir = os.path.join(self.init_params["model_dir"], self._params["model_id"].value["cur"])
		
				if self._params["model_id"].value["warm_start_from"] is not None:
					warm_start = os.path.join(
						self.init_params["model_dir"], 
						self._params["model_id"].value["warm_start_from"])
				else:
					warm_start = None
				

			except Exception as e:
				traceback.print_exc()
				print(self._params)
				model_dir = self.init_params["model_dir"] + str(uuid.uuid4())
				warm_start = None

		else:
			model_dir = self.init_params["model_dir"] + str(uuid.uuid4())
			warm_start = None

		self.estimator = tf.estimator.Estimator(
			model_fn=self.init_params["model_fn"],
			model_dir=model_dir,
			config=self.init_params.get("run_config", None),
			params=vars(self.friendly_params),
			warm_start_from=warm_start
		)

		self.trained = False

	def ensure_warm(self):
		if self.estimator is None:
			self.setup_estimator()

		# We need to warm up the estimator
		if not self.use_warm_start and not self.trained:
			self.do_step(1)


	def extract_vars(self):
		self.ensure_warm()
		var_names = self.estimator.get_variable_names()
		vals = {k:self.estimator.get_variable_value(k) for k in var_names}
		self._params["vars"] = VariableParam(vals)
		


	# --------------------------------------------------------------------------
	# Worker class stub impl
	# --------------------------------------------------------------------------

	def pre_params_get(self):
		if not self.use_warm_start:
			self.extract_vars()  

	def post_params_set(self):
		self.setup_estimator()

	
	def do_step(self, steps, heartbeat, should_continue):
		# We lazily initialise the estimator as during unpickling we may not have all the params
		if self.estimator is None:
			self.setup_estimator()

		self.estimator.train(self.init_params["train_input_fn"](self.friendly_params), steps=steps)

		# TODO: put heartbeat and should_continue into a hook
		heartbeat()

		self.trained = True
		
	def do_eval(self):
		self.ensure_warm()
		return self.estimator.evaluate(self.init_params["eval_input_fn"](self.friendly_params))


	# --------------------------------------------------------------------------
	# Pickling
	# --------------------------------------------------------------------------
	
	def __getstate__(self):
		return {
			"_params":          self.params,
			"results":          self.results,
			"id":               self.id,
			"current_count":    self.current_count,
			"total_count":      self.total_count,
		}

	def __setstate__(self, state):
		self.id             = state.get("id", uuid.uuid1())
		self.total_count    = state.get("total_count", 0)
		self.current_count  = state.get("current_count", 0)

		self.results        = state.get("results", {})
		self._params        = state.get("_params", {})

		self.estimator = None
		self.trained = False

