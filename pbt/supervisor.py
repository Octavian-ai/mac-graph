
import os.path
import time
import pickle
import math
import uuid
import traceback
import random
import yaml
import logging
logger = logging.getLogger(__name__)

from util import Ploty

from .specs import *
from .param import FixedParam
from .queue import QueueFactory
from util import FileWritey, FileReadie

class Supervisor(object):

	def __init__(self, args, param_spec, score, name_fn=None, reverse=False, gen_baseline_params=None):
		'''
		Args:
			- `reverse` True means a lower worker score is better (e.g. loss), False means higher is better (e.g. accuracy)

		'''
		self.args = args
		self.param_spec = param_spec
		self.score = score
		self.name_fn = name_fn if name_fn is not None else lambda x:str(x.id)
		self.reverse = reverse
		self.gen_baseline_params = gen_baseline_params

		self.workers = {}
		
		self.time_last_save = time.time()
		self.time_last_print = time.time()
		self.print_dirty = False

		self.plot_progress = Ploty(args, title='Training progress', x='Time', y="Value")
		self.plot_best_score = Ploty(args, title='Best score', x='Time', y="Value")

		self.plot_measures = {}
		self.measures = {
			"score": self.score
		}
		self.ensure_has_measure("score")

		self.queue_result = QueueFactory.vend(self.args, "result", "result_supervisor", "*")
		self.queue_run = QueueFactory.vend(self.args, "run", "run_shared", "run")
		
		if self.args.load:
			self.load()
		
	def load(self):
		logger.info("Trying to load workers from " + self.file_path)
	
		try:
			with FileReadie(self.args, "workers.pkl", True) as file:
				self.workers = pickle.load(file)
				logger.info("Loaded {} workers".format(len(self.workers)))
				
		except Exception:
			self.workers = {}

		self.time_last_save = time.time()

		self.plot_progress.load()
		self.plot_best_score.load()


	def save(self):
		logger.debug("Saving workers to " + self.file_path)

		with FileWritey(self.args, "workers.pkl", True) as file:
			pickle.dump(self.workers, file)

		with FileWritey(self.args, "workers.yaml", False) as file:
			yaml.dump(self.workers, file)
			
		self.time_last_save = time.time()


	# --------------------------------------------------------------------------
	# Core loop
	# --------------------------------------------------------------------------
	
	def run_epoch(self):
		self.scale_workers()
		self.dispatch_idle()
		self.consider_save()
		self.consider_print()
		self.get_messages()

	

	# --------------------------------------------------------------------------
	# Scale workers up and down
	# --------------------------------------------------------------------------

	def scale_workers(self):
		delta = self.args.n_workers - len(self.workers)

		if delta > 0:
			if self.args.run_baseline:
				baseline_set = self.gen_baseline_params()[len(self.workers) : len(self.workers)+delta]
				for i in baseline_set:
					logger.info("New worker from baseline set")
					self.add_worker(params=i)
			else:
				for i in range(delta):
					self.add_worker()

		elif delta < 0:
			for i in range(-delta):
				self.remove_worker()

	def generate_sexual(self):
		a = random.choice(self.workers.values())
		b = self.find_partner(a)
		logger.info("New worker from {}.breed({})".format(a.id,b.id))
		c = a.breed(b)
		return c

	def generate_asexual(self):
		mentor = self.find_mentor()
		logger.info("New worker from {}.mutate() total_steps:{}".format(mentor.id, mentor.total_steps))
		return mentor.mutate(self.args.heat)



	def add_worker(self, params=None, results=None):
		if params is None:
			try:
				if self.args.breed_sexual:
					newbie = self.generate_sexual()
				else:
					newbie = self.generate_asexual()

			except Exception as ex:
				logger.info("New worker from param spec realize ({})".format(str(ex)))
				params = self.param_spec.realize()
				newbie = WorkerHeader(params)
		else:
			newbie = WorkerHeader(params)
			newbie.results = results

		self.workers[newbie.id] = newbie
		self.dispatch(newbie)


	def remove_worker(self):
		if len(self.workers) > 0:
			stack = self.get_sorted_workers()
			if len(stack) < self.n_workers / 2:
				raise ValueError("Cannot remove_worker as not enough workers have scores yet")
			else:
				del self.workers[stack[0].id]	



	# --------------------------------------------------------------------------
	# The genetic part
	# --------------------------------------------------------------------------

	def find_mentor(self):
		stack = self.get_sorted_workers()
		
		n20 = max(round(len(self.workers) * self.args.exploit_pct), 1)
		top20 = stack[-n20:]

		# Dont clone a fresh clone
		top20 = [i for i in top20 if i.total_steps >= self.args.micro_step]

		if len(top20) > 0:
			return random.choice(top20)

		raise ValueError("No top workers have results yet")

	def consider_exploit(self, worker):
		if worker.recent_steps >= self.args.micro_step * self.args.macro_step:

			worker.recent_steps = 0
			stack = self.get_sorted_workers()

			try:
				idx = stack.index(worker)

				# if we have enough results
				if len(stack) > 1 and len(stack) > len(self.workers)/2: 
					nLower = max(len(stack) * self.args.exploit_pct,1)

					# proportionally cull the bottom tranche
					if idx < nLower:
						del self.workers[worker.id]
						logger.info("del {}".format(worker.id))

						self.add_worker() # dispatches the worker
						return

				else:
					logger.debug("Not enough workers ({}) with results to cull and add new workers".format(len(stack)))

			except ValueError:
				# It's ok we couldn't index that worker, it means they've no score yet
				pass

			self.dispatch(worker)

		else:
			logger.debug("Worker {} has not worked enough {} < {} to consider exploit".format(self.name_fn(worker), self.args.micro_step * self.args.macro_step, worker.recent_steps))


	def find_partner(self, worker):

		def suitable(w):
			return w.params is not None \
				and w.results is not None \
				and w.id != worker.id \
				and self.args.sexual_compatibility_mix < w.dist(worker) < self.args.sexual_compatibility_max \
				and self.score(w) is not None

		partners = [i for i in self.workers.values() if suitable(i)]
		sort(partners, key=self.score, reverse=self.reverse)
		partners = partners[-self.args.sexual_top_candidates:]

		return random.choice(partners)


	




	# --------------------------------------------------------------------------
	# Send tasks out for work
	# --------------------------------------------------------------------------

	def dispatch(self, worker):
		"""Request drone runs this worker"""
		run_spec = worker.gen_run_spec(self.args)
		self.queue_run.send(run_spec)
		logger.info('{}.dispatch({},{})'.format(worker.id, run_spec.macro_step, run_spec.micro_step))
		worker.time_last_updated = time.time()
		worker.time_last_dispatched = time.time()

	def dispatch_idle(self):
		for i in self.workers.values():
			if time.time() - i.time_last_updated > self.args.job_timeout:
				logger.warning('{}.dispatch_idle()'.format(i.id))
				self.dispatch(i)




	# --------------------------------------------------------------------------
	# Messaging queue logistics
	# --------------------------------------------------------------------------

	def _handle_result(self, spec, ack, nack):		
		if spec.worker_id in self.workers:
			i = self.workers[spec.worker_id]

			if isinstance(spec, HeartbeatSpec):
				i.time_last_updated = time.time()
				# logger.debug("{}.record_heartbeat({}, {}, {})".format(spec.worker_id, spec.from_hostname, spec.total_steps, spec.run_id, spec.tiebreaker))

			elif isinstance(spec, ResultSpec):
				if spec.success:
					if spec.total_steps > i.total_steps:
						i.update_from_result_spec(spec)
						logger.info("{}.record_result({})".format(spec.worker_id, spec))

						self.print_dirty = True
						self.print_worker_results(i)
						
						self.consider_exploit(i)
					else:
						logger.warning("{} received results for {} < current total_steps {}".format(spec.worker_id, spec.total_steps, i.total_steps))

				elif not self.args.run_baseline:
					logger.info("del {}".format(spec.worker_id))
					del self.workers[spec.worker_id]
					self.add_worker()

				else:
					self.dispatch(i)
			
			elif isinstance(spec, GiveUpSpec):
				pass

			else:
				logger.warning("Received unknown message type {}".format(type(spec)))
		else:
			logger.debug("{} worker not found for message {}".format(spec.worker_id, spec))

		# Swallow bad messages
		# The design is for the supervisor to re-send and to re-spawn drones
		ack()

	def get_messages(self):
		self.queue_result.get_messages(lambda spec, ack, nack: self._handle_result(spec, ack, nack))

	def close(self):
		self.queue_result.close()
		self.queue_run.close()







	# --------------------------------------------------------------------------
	# Helpers
	# --------------------------------------------------------------------------


	def get_sorted_workers(self):
		"""Workers for which no score is known will not be returned"""

		stack = [i for i in self.workers.values() if self.score(i) is not None]
		random.shuffle(stack)
		stack.sort(key=self.score, reverse=self.reverse)
		return stack

	def ensure_has_measure(self, key):
		def get_metric(worker):
			try:
				return worker.results[key]
			except Exception:
				return None

		if key not in self.measures:
			self.measures[key] = get_metric

		if key not in self.plot_measures:
			self.plot_measures[key] = Ploty(self.args, title="Metric "+key, x='Time', y=key)
			if self.args.load:
				self.plot_measures[key].load()


	def print_worker_results(self, worker):
		if worker.results is not None:

			name = self.name_fn(worker)

			for key in worker.results:
				self.ensure_has_measure(key)

			for key in self.measures:
				val = self.measures[key](worker)
				if val is not None:
					plot = self.plot_measures[key]
					plot.add_result(time.time(), val, name)
					plot.write()
			

	def print(self):
		logger.debug("Printing worker performance")

		def plot_param_metrics(plot, worker, prefix="", suffix=""):
			for key, val in worker.params.items():
				if not isinstance(val, FixedParam):
					if isinstance(val.metric, int) or isinstance(val.metric, float):
						if val.metric is not None:
							plot.add_result(time.time(), val.metric, prefix+key+suffix)
					elif isinstance(val.metric, dict):
						for mkey, mval in val.metric.items():
							if isinstance(mval, int) or isinstance(mval, float):
								if mval is not None:
									plot.add_result(time.time(), mval, prefix+key+"_"+mkey+suffix)


		stack = self.get_sorted_workers()

		for key, fn in self.measures.items():
			vs = [fn(i) for i in self.workers.values() if fn(i) is not None]

			if len(vs) > 0:
				best = max(vs)
				worst = min(vs)
				self.plot_progress.add_result(time.time(), best, key+"_max")
				self.plot_progress.add_result(time.time(), worst, key+"_min")

		self.plot_progress.add_result(time.time(), len(self.workers), "n_workers")

		if len(stack) > 0:
			best_worker = stack[-1]
			plot_param_metrics(self.plot_progress, best_worker, suffix="_best")

			self.plot_best_score.add_result(time.time(), self.score(best_worker), "score")

		self.plot_progress.write()
		self.plot_best_score.write()

		self.print_dirty = False

	def consider_save(self):
		if time.time() - self.time_last_save > self.args.save_secs:
			self.save()

	def consider_print(self):
		if time.time() - self.time_last_print > self.args.print_secs and self.print_dirty:
			self.print()

	@property
	def file_path(self):
		return os.path.join(self.args.output_dir, self.args.run, "workers.pkl")

	





