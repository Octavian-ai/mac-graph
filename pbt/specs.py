
import uuid
import time
import collections
import platform
import petname

class BaseSpec(object):
	def __init__(self, group):
		self.id = uuid.uuid4()
		self.from_hostname = platform.node()
		self.time_sent = time.time()
		self.group = group



RunSpec = collections.namedtuple('RunSpec', [
	'id',
	'group', 
	'worker_id', 
	'from_hostname',
	'params',
	'recent_steps',
	'total_steps',
	'micro_step',
	'macro_step',
	'time_sent'])


ResultSpec = collections.namedtuple('ResultSpec', [
	'group', 
	'worker_id', 
	'from_hostname',
	'results', 
	'success', 
	'steps', 
	'recent_steps',
	'total_steps',
	'params',
	'time_sent'])

HeartbeatSpec = collections.namedtuple('HeartbeatSpec', [
	'group', 
	'from_hostname',
	'run_id',
	'worker_id', 
	'tiebreaker',
	'total_steps',
	'time_sent'])

GiveUpSpec = collections.namedtuple('GiveUpSpec', [
	'group',
	'from_hostname',
	'time_sent',
	'run_id',
	'worker_id'
])



class WorkerHeader(object):

	def __init__(self, params):
		self.id = petname.Generate(3,'-') + "-" + str(uuid.uuid4())
		self.params = params

		self.results = None
		self.time_last_updated = 0
		self.total_steps = 0
		self.recent_steps = 0


	def update_from_result_spec(self, result_spec):
		self.total_steps = result_spec.total_steps
		self.recent_steps = result_spec.recent_steps
		self.results = result_spec.results
		self.time_last_updated = time.time()
		self.time_last_dispatched = 0

		# TODO: impl properly
		# Protecting the params, it should in theory be fine to copy all of them over
		# self.params["model_id"] = result_spec.params["model_id"]

	def gen_run_spec(self, args):
		return RunSpec(
			uuid.uuid4(),
			args.run, 
			self.id, 
			platform.node(),
			self.params, 
			self.recent_steps,
			self.total_steps,
			args.micro_step, 
			args.macro_step,
			time.time()
		)

	def dist(self, other):
		return self.params.dist(other.params)

	def breed(self, other, heat):
		params = self.params.breed(other.params, heat)
		return WorkerHeader(params)

	def mutate(self, heat):
		params = self.params.mutate(heat)
		return WorkerHeader(params)




