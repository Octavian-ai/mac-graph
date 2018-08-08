
import traceback
import argparse
import os.path
import os

import tensorflow as tf
import numpy as np

import coloredlogs
from util.stackdriver import install as install_stackdriver

import logging
logger = logging.getLogger(__name__)

from .args import get_args
from macgraph.model import model_fn
from macgraph.input import gen_input_fn
from macgraph import get_args as get_macgraph_args
from macgraph import generate_args_derivatives

from pbt import *

def gen_param_spec(args):

	mga = get_macgraph_args(argv=[])

	p = {
		"heritage": Heritage,
		"model_id": ModelId,

		"vocab_size": RandIntParamOf(110,512),
		"max_decode_iterations": IntParamOf(4, 1, 32),
		"learning_rate": LRParam,
	}

	for i in ["embed_width", "memory_width", "control_width"]:
		p[i] = RandIntParamOf(4, 32) # make this 2048


	for i in ["memory_transform_layers", "output_layers", "input_layers", "control_heads"]:
		p[i] = RandIntParamOf(1, 8)

	return ParamSpec(p)



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

def install_logging(args):
	loggers = [logging.getLogger(i) 
		for i in ["__main__", "pbt", "experiment", "macgraph", "util", "tensorflow"]]

	for i in loggers:
		i.handlers = []
		if args.log_format == 'colored':
			coloredlogs.install(logger=i, level=args.log_level)
		elif args.log_format == 'json':
			install_stackdriver(logger=i, level=args.log_level)


