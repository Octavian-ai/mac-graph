

from .worker import Worker
from .param import *
from .params import *


def mock_hyperparam_spec():
	return ParamSpec({
		"micro_step": FixedParamOf(1),
		"macro_step": FixedParamOf(1),
		"state": FixedParamOf(0)
	})

class MockArgs(object):
	def __init__(self):
		self.output_dir = "./output_test/supervisor_test"
		self.bucket = None
		self.gcs_dir = None
		self.single_threaded = False
		self.save = True
		self.load = True
		self.n_workers = 10
		self.floyd_metrics = False
		self.queue_type = "rabbitmq"
		self.run = "default"
		self.run_baseline = False
		self.breed_sexual = False
		self.micro_step = 1
		self.macro_step = 1
		self.amqp_url = "amqp://guest:guest@localhost"
		self.message_timeout = 60

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