
import tensorflow as tf
import random
import pickle
import uuid
import os
import os.path
import collections
import logging
import sys
import math
import time

# from comet_ml import Experiment

from .params import MergedParams

import logging
logger = logging.getLogger(__name__)


class Worker(object):
	"""Runs a PBT experiment

	Always provide a parameterless init so the Supervisor can spawn workers as needed

	"""
	def __init__(self, init_params, params):
		
		# self.experiment = Experiment(api_key="bRptcjkrwOuba29GcyiNaGDbj")
		self.id = uuid.uuid4()
		
		self.init_params = init_params
		self.params = params

		self.results = {}
		self.total_steps = 0
		self.recent_steps = 0
		self.time_started = 0

	def update_from_run_spec(self, run_spec):
		self.params = run_spec.params
		self.total_steps = run_spec.total_steps
		self.recent_steps = run_spec.recent_steps

	# --------------------------------------------------------------------------
	# Implement these
	# --------------------------------------------------------------------------

	def do_step(self, steps, heartbeat, should_continue):
		"""Execute a training step. Returns nothing.

			:param heartbeat Call this function each training iteration, so the supervisor knows you're alive
			:param should_continue Call this function each training iteration and continue if it returns true
		"""
		pass

	def do_eval(self):
		"""Returns evaluation results as a dict"""
		pass


	def pre_params_get(self):
		'''Hook for doing setup before params can be read'''
		pass

	def post_params_set(self):
		'''Hook for doing post-processing on new params'''
		pass



	# --------------------------------------------------------------------------
	# Parameter methods
	# --------------------------------------------------------------------------

	@property
	def params(self):
		self.pre_params_get()
		return self._params
	
	@params.setter
	def params(self, params):
		self._params = params
		self.post_params_set()

	# Experimental, plan to roll this out everywhere to replace params
	@property
	def friendly_params(self):
		return MergedParams(self.init_params, self._params)
		

	# --------------------------------------------------------------------------
	# Properties
	# --------------------------------------------------------------------------
	

	# Will crash if model_id param missing
	# @returns dirictory string to save model to
	@property
	def model_dir(self):
		if self.friendly_params["model_id"]["cur"] is None:
			return None
		else:
			return os.path.join(self.init_params["model_dir"], self.init_params["run"], self.friendly_params["model_id"]["cur"])

	
	# Will crash if model_id param missing	
	# @returns directory string to warm start model from or None if model should not warm start	
	@property
	def warm_start_dir(self):
		if self.friendly_params["model_id"]["warm_start_from"] is None:
			return None
		else:
			return os.path.join(self.init_params["model_dir"], self.init_params["run"], self.friendly_params["model_id"]["warm_start_from"])

	# --------------------------------------------------------------------------
	# Step and eval
	# --------------------------------------------------------------------------
		
	def step(self, steps, heartbeat, should_continue):
		started = time.time()
		
		self.do_step(steps, heartbeat, should_continue)

		self.recent_steps += steps
		self.total_steps += steps

		time_taken = time.time() - started
		tf.logging.info("train_op/second: {}".format(float(steps)/float(time_taken)))


	def eval(self):
		self.results = self.do_eval()
		return self.results


	def step_and_eval(self, steps, heartbeat, should_continue):
		logger.info("{}.train({})".format(self.id, steps))
		self.step(steps, heartbeat, should_continue)
		should_continue()
		logger.info("{}.eval()".format(self.id))
		return self.eval()


	# --------------------------------------------------------------------------
	# Load and save
	# --------------------------------------------------------------------------

	def save(self, path):
		with open(path, 'wb') as file:
			pickle.dump(self, file)

	@classmethod
	def load(cls, path, init_params):
		with open(path, 'rb') as file:
			w = pickle.load(file)
		w.init_params = init_params
		return w

	 