\
import traceback
import pickle
import time
import collections
import platform
import uuid
import random
import petname

import logging
logger = logging.getLogger(__name__)

# Hack for single-threaded
from .specs import *
from .queue import QueueFactory

Perf = collections.namedtuple('Perf', ['time_start', 'time_end', 'steps'])

class Drone(object):
	
	def __init__(self, args, SubjectClass, init_params):
		self.args = args
		self.SubjectClass = SubjectClass
		self.init_params = init_params
		self.worker_cache = {}
		self.run_max = {}
		self.drone_id = petname.Generate(3, '-') #uuid.uuid1()

		self.logger = logging.getLogger(__name__ + "." + self.drone_id)

		self.performance = []
		self.steps_per_sec = 0
		self.time_last_heartbeat = 0
		self.time_last_token_check = 0

		self.queue_result = QueueFactory.vend(self.args, "result", "result_shared", "result")
		self.queue_heartbeat = QueueFactory.vend(self.args, "result", str(self.drone_id), "heartbeat")
		self.queue_run = QueueFactory.vend(self.args, "run", "run_shared", "run")


	def _send_result(self, run_spec, worker, success):
		result_spec = ResultSpec(
			self.args.run, 
			run_spec.worker_id, 
			platform.node(),
			worker.results, 
			success, 
			run_spec.micro_step,
			worker.recent_steps,
			worker.total_steps, 
			worker.params,
			time.time())

		self.queue_result.send(result_spec)

	def _send_heartbeat(self, worker, run_spec, tiebreaker):
		if time.time() - self.time_last_heartbeat > self.args.job_timeout / 6:
			spec = HeartbeatSpec(
				self.args.run, 
				platform.node(),
				run_spec.id,
				worker.id, 
				tiebreaker, 
				worker.total_steps,
				time.time())

			self.queue_heartbeat.send(spec)
			self.time_last_heartbeat = time.time()

	def _handle_heartbeat(self, spec):
		cur = self.run_max.get(spec.worker_id, 0)
		self.run_max[spec.worker_id] = max(cur, spec.total_steps + spec.tiebreaker)

	def _should_continue(self, worker, run_spec, tiebreaker):
		if time.time() - self.time_last_token_check > self.args.job_timeout / 6:
			self.queue_heartbeat.get_messages(lambda m, ack, nack: self._handle_heartbeat(m))
			self.time_last_token_check = time.time()

		try:
			should = self.run_max[run_spec.worker_id] <= tiebreaker + worker.total_steps
			if not should:
				self.queue_result.send(GiveUpSpec(
					self.args.run,
					platform.node(),
					time.time(),
					run_spec.id,
					worker.id					
				))
				self.logger.info("Should not continue, heard another heartbeat with higher run")
				raise StopIteration()
			
		except KeyError:
			pass

		return True


	def _handle_run(self, run_spec):

		if run_spec.worker_id in self.worker_cache:
			worker = self.worker_cache[run_spec.worker_id]
		else:
			worker = self.SubjectClass(self.init_params, run_spec.params)
			worker.id = run_spec.worker_id
			self.logger.debug("{} init with params {}".format(worker.id, run_spec.params))
			self.worker_cache[run_spec.worker_id] = worker

		worker.update_from_run_spec(run_spec)

		# Good idea, bad impl
		# Stop multiple workers over-writing eachother's saves
		# worker.params["model_id"] = worker.params["model_id"].mutate(0)
		
		try:
			time_start = time.time()
			self.logger.info("{}.step_and_eval({}, {})".format(run_spec.worker_id, run_spec.macro_step, run_spec.micro_step))
			
			self.time_last_heartbeat = 0
			self.time_last_token_check = 0
			tiebreaker = random.random()

			send_heartbeat  = lambda:self._send_heartbeat(worker, run_spec, tiebreaker)
			should_continue = lambda:self._should_continue(worker, run_spec, tiebreaker)

			try:
				for i in range(run_spec.macro_step):
					worker.step_and_eval(run_spec.micro_step, send_heartbeat, should_continue)
					self._send_result(run_spec, worker, True)
			except StopIteration:
				pass

			self.performance.append(Perf(time_start, time.time(), run_spec.micro_step * run_spec.macro_step))
			self.print_performance()
		except Exception as e:
			traceback.print_exc()
			self._send_result(run_spec, worker, False)
		

	def print_performance(self):
		window = 60 * 5
		cutoff = time.time() - window
		perf = [i for i in self.performance if i.time_start >= cutoff]

		if len(perf) > 0:
			start = min([i.time_start for i in perf])
			end = max([i.time_end for i in perf])

			duration = end - start
			steps = sum([i.steps for i in perf])

			self.steps_per_sec = steps / duration
			self.logger.info("Steps per second: {}".format(self.steps_per_sec))



	def get_messages(self):
		self.queue_run.get_messages(lambda data, ack, nack: self._handle_run(data), 1)
		

	def run_epoch(self):
		self.get_messages()

	def close(self):
		self.queue_run.close()
		self.queue_result.close()



