
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

		def score_fn(worker):
			try:
				return worker.results["accuracy"]
			except Exception:
				return None

		s = Supervisor(
			MockArgs(),
			mock_hyperparam_spec(),
			score_fn,
		)

		return s

	def test_scale_workers(self):
		s = self.vend_supervisor()
		s.scale_workers()
		s.args.n_workers = 3
		s.scale_workers()
		s.args.n_workers = 10
		s.scale_workers()



if __name__ == '__main__':	
	unittest.main()




