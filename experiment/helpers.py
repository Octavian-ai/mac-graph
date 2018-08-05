
import traceback
import argparse
import os.path
import os

import tensorflow as tf
import numpy as np

import logging
logger = logging.getLogger(__name__)

from .args import get_args
from macgraph.model import model_fn 

from pbt import *

def gen_param_spec(args):
	return ParamSpec({
		"heritage": Heritage,
		"model_id": ModelId,
	})

def gen_worker_init_params(args):
	
	p = {
		"model_fn": model_fn, 
		"train_input_fn": gen_input_fn(args, "train"), 
		"eval_input_fn":  gen_input_fn(args, "eval"),
		"run_config": tf.estimator.RunConfig(save_checkpoints_steps=99999999999,save_checkpoints_secs=None)
	}

	p.update(vars(args))

	return p

def get_drone(args):
	return Drone(args, EstimatorWorker, gen_worker_init_params(args))


def score(worker):
	try:
		# return (worker.results["loss"] + 1) / worker.results["total_elements"]
		return worker.results["correct_elements"] - worker.results["loss"]/10
	except Exception:
		return None

def name_fn(worker):
	return worker.params["heritage"].value + "_" + str(worker.id)[-5:-1]


def get_supervisor(args):
	return Supervisor(args, gen_param_spec(args), score, name_fn, False)






