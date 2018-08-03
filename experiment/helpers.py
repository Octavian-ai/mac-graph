
import traceback
import argparse
import os.path
import os

import tensorflow as tf
import numpy as np

import logging
logger = logging.getLogger(__name__)

from .args import get_args
from macgraph import *

from pbt import *

#from dnc import *
#
## Ideally this should be wrapped up into an args structure, but I've yet to think out the plumbing
#MAX_LENGTH  = int(os.getenv("MAX_LENGTH", 2))
#MAX_REPEATS = int(os.getenv("MAX_REPEATS", 5))

#class DatasetParam(GeneticParam):
#
#	def __init__(self, batch_size, v=None):
#
#		self.batch_size = batch_size
#		self.v = v
#
#		if self.v is None:
#			self.v = {
#				"length": RandIntRangeParamOf(1, MAX_LENGTH)(),
#				"repeats": RandIntRangeParamOf(1, MAX_REPEATS)(),
#			}
#
#
#	def _mutate_dict(self, heat):
#		return {
#			k: val.mutate(heat)
#			for k, val in self.v.items()
#		}
#
#	def mutate(self, heat):
#		return type(self)(self.batch_size, self._mutate_dict(heat))
#
#	@property
#	def name_str(self):
#		return "l"+ str(self.v["length"].value) + "r" + str(self.v["repeats"].value)
#
#
#	@property
#	def value(self):
#		return repeat_copy.RepeatCopy(
#			num_bits=4, 
#			batch_size=self.batch_size, 
#			**self.metric
#		)
#
#	def __eq__(self, other):
#		self.v == other.v
#
#	@property
#	def metric(self):
#		return {
#			"min_length":  self.v["length"].value[0],
#			"max_length":  self.v["length"].value[1],
#			"min_repeats": self.v["repeats"].value[0],
#			"max_repeats": self.v["repeats"].value[1],
#		}
#
#
#def DatasetParamOf(batch_size):
#	def m(v=None):
#		return DatasetParam(batch_size, v)
#	return m

#def gen_dataset_eval(args):
#	return lambda: DatasetParam(args.batch_size, {
#		"length":  RangeParam([FixedParam(MAX_LENGTH), FixedParam(MAX_LENGTH)]),
#		"repeats": RangeParam([FixedParam(MAX_REPEATS), FixedParam(MAX_REPEATS)])
#	})
#
def gen_param_spec(args):
	return ParamSpec({
		"heritage": Heritage,
		"model_id": ModelId,
	})


#def gen_input_fn(is_eval=False):
#	def g(params):
#		def gen():
#			seed = 123 if is_eval else None
#			src = "dataset_eval" if is_eval else "dataset_train"
#			dataset_tensors = params[src](seed)
#
#			return (
#				dataset_tensors.observations, 
#				{
#					"target": dataset_tensors.target,
#					"mask": dataset_tensors.mask,
#					"length": dataset_tensors.length,
#					"total_targ_batch": dataset_tensors.total_targ_batch,
#				}
#			)
#
#		return gen
#	return g

def gen_worker_init_params(args):
	
	p = {
		"model_fn": model_fn, 
		"train_input_fn": gen_input_fn(args, "train"), 
		"eval_input_fn":  gen_input_fn(args, "eval"),
		"eval_steps": 20,
		"run_config": tf.estimator.RunConfig(save_checkpoints_steps=99999999999,save_checkpoints_secs=None)
	}

	p.update(vars(args))

	return p

def get_drone(args):
	return Drone(args, SingularSessionWorker, gen_worker_init_params(args))


def score(worker):
	try:
		# return (worker.results["loss"] + 1) / worker.results["total_elements"]
		return worker.results["correct_elements"] - worker.results["loss"]/10
	except Exception:
		return None

def name_fn(worker):
	return worker.params["dataset_train"].name_str + "_" + worker.params["heritage"].value + "_" + str(worker.id)[-5:-1]

def gen_baseline_params(args):

	def g():
		"""This is a set of params for generating the 'baseline' workers, e.g. the reference
		   that this experiment is trying to out-perform
		   """

		param_spec = gen_param_spec(args)

		lengths = [pow(2,i) for i in range(0, 6) if pow(2,i) <= MAX_LENGTH]
		repeats = [pow(2,i) for i in range(0, 6) if pow(2,i) <= MAX_REPEATS]

		datasets = []

		for i in lengths:
			for j in repeats:
				datasets.append(DatasetParam(args.batch_size, {
					"length":  RangeParam([FixedParam(i), FixedParam(MAX_LENGTH)]),
					"repeats": RangeParam([FixedParam(j), FixedParam(MAX_REPEATS)]),
				}))

		param_sets = []
		for i in datasets:
			params = param_spec.realize()
			params["dataset_train"] = i
			param_sets.append(params)

		return param_sets

	return g

def get_supervisor(args):
	return Supervisor(args, gen_param_spec(args), score, name_fn, False, gen_baseline_params=gen_baseline_params(args))






