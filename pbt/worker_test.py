
import unittest
import os
import os.path
import sys
import pathlib

import logging
logger = logging.getLogger(__name__)

output_dir = "./output_test/worker_test"

from .mock import *

class WorkerTestCase(unittest.TestCase):
	"""This test case is designed so you can easily use it to test worker implementations"""

	# --------------------------------------------------------------------------
	# Implement these to use this test
	# --------------------------------------------------------------------------

	def vend_worker(self):
		return MockWorker()

	def load_worker(self, file_path):
		return MockWorker.load(file_path, {})

	@property
	def steps(self):
		return 1
	
	


	# --------------------------------------------------------------------------
	# Helpers
	# --------------------------------------------------------------------------
	
	def assertDictAlmostEqual(self, first, second, threshold=0.01, msg=None):
		for key, val in first.items():
			delta = abs(float(val) - float(second[key]))
			pct = delta / (float(val) + 0.00000001)
			self.assertTrue(pct < threshold, key + ": " + msg)
			# self.assertAlmostEqual(val, second[key], places, key + ": " + msg)


	def assertDictEqual(self, first, second, msg=None):
		for key, val in first.items():
			self.assertEqual(first[key], second[key], "{}: {}".format(key, msg))


	def setUp(self):
		try:
			pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True) 
		except Exception:
			pass


	
	# ==========================================================================
	# Tests
	# ==========================================================================
	
	def test_save_load(self):
		file_path = os.path.join(output_dir, "save_load.pkl")

		try:
			os.unlink(file_path)
		except Exception:
			pass

		worker1 = self.vend_worker()
		worker1.step(self.steps)
		worker1.eval()
		worker1.save(file_path)
		worker2 = self.load_worker(file_path)

		self.assertDictEqual(worker1.results, worker2.results)
		self.assertDictEqual(worker1.params, worker2.params)

		worker2.eval()

		self.assertDictEqual(worker1.results, worker2.results, msg="Evaluation after loading and eval should be unchanged")
		self.assertDictEqual(worker1.params, worker2.params)


	def test_param_copy(self):
		worker1 = self.vend_worker()
		worker1.step(self.steps)
		worker1.eval()

		logger.info("-------------- NOW TRANSFER PARAMS 1 --------------")

		worker2 = self.vend_worker()
		worker2.params = worker1.explore(0.0)
		worker2.eval()

		self.assertDictEqual(worker1.results, worker2.results, "Results should be equal after param explore copy")

		logger.info("-------------- NOW TRANSFER PARAMS 2 --------------")

		worker3 = self.vend_worker()
		worker3.params = worker1.params
		worker3.eval()

		self.assertDictEqual(worker1.results, worker3.results, "Results should be equal after param copy")


	def test_mutate(self):

		worker = self.vend_worker()

		for i in range(10):
			worker.step(self.steps)
			worker.eval()
			worker.params = worker.explore(1.0)

		# Didn't crash = success


