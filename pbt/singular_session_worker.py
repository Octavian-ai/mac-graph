
import tensorflow as tf
import numpy as np

import traceback
import os.path
import time

from .worker import Worker
from .param import *
from .params import *
from util import path_exists

import logging
logger = logging.getLogger(__name__)

tf.logging.set_verbosity("INFO")

class ModelSession(object):
	def __init__(self, init_params, friendly_params, model_dir, warm_start_dir, mode):

		self.init_params = init_params
		self.friendly_params = friendly_params
		self.model_dir = model_dir
		self.warm_start_dir = warm_start_dir

		self.graph = tf.Graph()
		with self.graph.as_default():

			input_fn = self.init_params[mode+"_input_fn"](self.friendly_params)
			inpt = input_fn()

			self.model = self.init_params["model_fn"](
				inpt[0],
				inpt[1],
				mode,
				self.friendly_params
			)

			self.model_mode = mode

			hooks = [
			]

			if mode == "train":
				self.saver = tf.train.Saver()
				self.checkpoint_saver = tf.train.CheckpointSaverHook(
						checkpoint_dir=self.model_dir,
						save_steps=self.friendly_params["micro_step"],
						saver=self.saver)

				hooks.append(self.checkpoint_saver)

			if self.init_params.get('profile', False):
				profiler = tf.train.ProfilerHook(save_steps=100, output_dir=self.model_dir)
				hooks.append(profiler)


			if tf.summary.merge_all() is not None:
				hooks.append(
					tf.train.SummarySaverHook(
						save_secs=10, 
						output_dir=self.model_dir, 
						summary_op=tf.summary.merge_all()
					)
				)

			# Transparent across GCS and local paths
			if path_exists(self.model_dir):
				# We should resume from that location
				load_dir = self.model_dir
			else:
				# We should try to warm start
				load_dir = self.warm_start_dir

			logger.debug("model_dir: {}  warm_start_dir: {} load from:{}".format(self.model_dir, self.warm_start_dir, load_dir))

			self.sess = tf.train.SingularMonitoredSession(
				hooks=hooks, checkpoint_dir=load_dir
			)

	def run(self, ops):
		with self.graph.as_default():
			return self.sess.run(ops)

	def close(self):
		if self.sess is not None:
			self.sess.close()
			self.sess = None

		self.graph = None
		self.model = None

	def __enter__(self):
		return self

	def __exit__(self, a, b, c):
		self.close()

class SingularSessionWorker(Worker):
	
	def __init__(self, init_params, hyperparam_spec):
		self._params = {}

		super().__init__(init_params, hyperparam_spec)


	def get_model_session(self, mode="train"):
		return ModelSession(
			self.init_params,
			self.friendly_params,
			self.model_dir,
			self.warm_start_dir,
			mode
		)


		
	def do_step(self, steps, heartbeat, should_continue):
		with self.get_model_session("train") as sm:
			for i in range(steps):
				_, loss = sm.run([sm.model.train_op, sm.model.loss])
				heartbeat()
				should_continue()

			

	def do_eval(self):
		with self.get_model_session("eval") as sm:
			for i in range(self.init_params["eval_steps"]):
				r = sm.run(sm.model.eval_metric_ops)

		return {
			k: float(v[0])
			for k, v in r.items()
		}
		

	# Hooks for Pickle
	def __getstate__(self):
		return {
			"_params":          self.params,
			"results":          self.results,
			"id":               self.id,
			"current_count":    self.current_count,
			"total_count":      self.total_count,
		}

	def __setstate__(self, state):
		self.id             = state.get("id", uuid.uuid4())
		self.time_started 	= 0
		self.performance 	= (0,0)
		
		self.total_count    = state.get("total_count", 0)
		self.current_count  = state.get("current_count", 0)

		self.results        = state.get("results", {})
		self._params        = state.get("_params", {})

		

