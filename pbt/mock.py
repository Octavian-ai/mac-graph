

from .worker import Worker
from .param import *


def mock_hyperparam_spec():
	return {
		"micro_step": FixedParamOf(1),
		"macro_step": FixedParamOf(1),
		"state": FixedParamOf(0)
	}

class MockArgs(object):
	def __init__(self):
		self.output_dir = "./output_test/supervisor_test"
		self.bucket = None
		self.gcs_dir = None
		self.single_threaded = False
		self.save = True
		self.load = True

class MockWorker(Worker):
	"""For use in tests"""

	def __init__(self, init_params={}, hyperparam_spec=None):

		if hyperparam_spec is None:
			hyperparam_spec = mock_hyperparam_spec()

		super().__init__(init_params, hyperparam_spec)

	def do_eval(self):
		return {
			"accuracy": self.friendly_params["state"]
		}

	def do_step(self, steps):
		self._params["state"].v += 1