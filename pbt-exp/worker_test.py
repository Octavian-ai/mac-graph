
import unittest
import tensorflow as tf

import pbt

from .args import get_args
from .helpers import gen_param_spec, gen_worker_init_params, DatasetParam

import logging
logger = logging.getLogger(__name__)

class WorkerTestCase(pbt.WorkerTestCase):

	def vend_worker(self):
		args = get_args([])
		init_params = gen_worker_init_params(args)
		hyperparam_spec = gen_param_spec(args)
		return pbt.SingularSessionWorker(init_params, hyperparam_spec)

	def load_worker(self, file_path):
		args = get_args([])
		init_params = gen_worker_init_params(args)
		return pbt.SingularSessionWorker.load(file_path, init_params)

	@property
	def steps(self):
		return 2000
	


	def assertDatasetEqual(self, first, second, msg):
		self.assertEqual(first.v, second.v, msg)

	def setUp(self):
		self.addTypeEqualityFunc(DatasetParam, self.assertDatasetEqual)
		
	



if __name__ == '__main__':	
	# tf.logging.set_verbosity('INFO')
	# logger.setLevel('INFO')
	unittest.main()


