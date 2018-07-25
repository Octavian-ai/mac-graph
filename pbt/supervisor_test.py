
import unittest
import os
import os.path
import sys
import pathlib

import logging
logger = logging.getLogger(__name__)

from .supervisor import Supervisor
from .worker import Worker
from .param import *

from .mock import *

class SupervisorTestCase(unittest.TestCase):

	def vend_supervisor(self):
		s = Supervisor(
			MockArgs(),
			MockWorker,
			{},
			mock_hyperparam_spec(),
			lambda worker: worker.results["accuracy"],
			n_workers = 10,
		)

		return s

	def test_run(self):
		s = self.vend_supervisor()
		s.run(1)


	def test_load_save(self):
		s = self.vend_supervisor()
		self.assertEqual(len(s), 0)
		s.run(1)
		self.assertEqual(len(s), 10)
		s.save()
		self.assertEqual(s.best_worker.results, {"accuracy":1})

		p = self.vend_supervisor()
		self.assertEqual(len(p), 0)
		p.load()
		self.assertEqual(len(p), 10)
		self.assertEqual(s.best_worker.results, p.best_worker.results)
		p.run(1)
		self.assertEqual(len(p), 10)
		p.save()
		self.assertEqual(len(p), 10)
		self.assertNotEqual(s.best_worker.results, p.best_worker.results)
		



if __name__ == '__main__':	
	# tf.logging.set_verbosity('INFO')
	# logger.setLevel('INFO')
	unittest.main()




