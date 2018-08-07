
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
from macgraph.input import gen_input_fn
from macgraph import get_args as get_macgraph_args
from macgraph import generate_args_derivatives

from pbt import *

def gen_param_spec(args):
    return ParamSpec({
        "heritage": Heritage,
        "model_id": ModelId,
<<<<<<< Updated upstream
        "vocab_size":  IntParamOf(128, 4, 2048),
		"embed_width": IntParamOf(64, 4, 2048),
		"learning_rate": LRParam,
=======
        "vocab_size":  IntParam,
		"embed_width": IntParam,
		"learning_rate": LRParam
>>>>>>> Stashed changes
    })



def gen_worker_init_params(args):

	p = {}

	# Get the defaults for macgraph arguments 
	# These will likely be overrided by ParamSpec items
	# and can be overrided by command line args from PBT
	p.update(get_macgraph_args(argv=[]))

	# Add all command line args from PBT
	p.update(vars(args))

	p.update(generate_args_derivatives(p))
	
	# Key pieces for Estimator
	p.update({
		"model_fn": model_fn,
		"train_input_fn": lambda params: gen_input_fn(p, "train"),
		"eval_input_fn":  lambda params: gen_input_fn(p, "eval"),
		"run_config": tf.estimator.RunConfig(save_checkpoints_steps=99999999999, save_checkpoints_secs=None)
	})

	return p

def get_drone(args):
    return Drone(args, EstimatorWorker, gen_worker_init_params(args))


def score(worker):
	try:
		return worker.results["loss"]
	except Exception:
		return None

def name_fn(worker):
	return worker.params["heritage"].value + "_" + str(worker.id)[-5:-1]


def get_supervisor(args):
    return Supervisor(args, gen_param_spec(args), score, name_fn, True)
