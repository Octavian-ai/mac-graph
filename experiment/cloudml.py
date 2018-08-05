# https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/census/tensorflowcore/trainer/task.py

import os
import tensorflow as tf
import json

import logging
logging.basicConfig()

logger = logging.getLogger(__name__)

from .helpers import *

def get_config():

	tf_config = os.environ.get('TF_CONFIG')
	
	# If TF_CONFIG is not available run local
	if not tf_config:
		return None

	tf_config_json = json.loads(tf_config)

	cluster = tf_config_json.get('cluster')
	job_name = tf_config_json.get('task', {}).get('type')
	task_index = tf_config_json.get('task', {}).get('index')
	is_chief = (job_name == 'master')

	cluster_spec = tf.train.ClusterSpec(cluster)
	server = tf.train.Server(cluster_spec,
		job_name=job_name,
		task_index=task_index)

	return job_name, task_index, cluster_spec, server


if __name__ == '__main__':

	config = get_config()

	if config is None:
		logger.warn("No TF_CONFIG env var")
		exit(-1)

	logger.info("Job name {}".format(config[0]))

	args = get_args()
	supervisor = get_supervisor(args)

	if args.load:
		supervisor.load()

	if config[0] == 'master':
		supervisor.manage()
	else:
		supervisor.drone()

	
