
import random
import numpy as np
import tensorflow as tf
import math
import string
import copy
import uuid
import collections

import logging
logger = logging.getLogger(__name__)

from util import path_exists

FP = collections.namedtuple('FallbackParam', ['value'])

class GeneticParam(object):
	"""Represents a parameter that can be sampled, copied, compared and mutated"""

	def __init__(self):
		"""Initialise with randomly sampled value"""
		pass

	def mutate(self, heat=1.0):
		"""Return copy of this param with mutated value"""
		pass

	def __eq__(self, other):
		return self.value == other.value

	def __str__(self):
		return str(self.value)

	def __repr__(self):
		return str(self.v)

	@property
	def value(self):
		return self.v

	@property
	def metric(self):
		"""Used for graphing hyper-parameters"""
		return self.value

	def dist(self, other):
		return np.linalg.norm([self.value - other.value])



class InitableParam(GeneticParam):
	def __init__(self,v=None):
		self.v = v

class NaturalNumbersParam(GeneticParam):
	def __init__(self,v=None):
		self.v = 1.0 if v is None else v

	def mutate(self, heat=1.0):
		return type(self).__init__(self.v + 1.0)


class FixedParam(InitableParam):
	def mutate(self, heat=1.0):
		return self

	def __str__(self):
		return ""

def FixedParamOf(v):
	return lambda: FixedParam(v)








class MulParam(InitableParam):
	def __init__(self, v, min, max):
		self.v = v
		self.max = max
		self.min = min

	@property
	def value(self):
		return min(max(self.v, self.min), self.max)

	def mutate(self, heat=1.0):
		return type(self)(self.value * random.uniform(0.8/heat, 1.2*heat) + random.gauss(0.0, heat*0.1), self.min, self.max)

def MulParamOf(v, min=-10000, max=10000):
	return lambda: MulParam(v, min, max)



class IntParam(MulParam):
	@property
	def value(self):
		return round(min(max(self.v, self.min), self.max))

def IntParamOf(v, min=1, max=1000):
	return lambda: IntParam(v, min, max)

def RandIntParamOf(min, max):
	return lambda: IntParam(random.randint(min, max), min, max)


class RangeParam(GeneticParam):
	def __init__(self, v):
		self.v = v

	@property
	def value(self):
		values = [i.value for i in self.v]
		return (min(values), max(values))

	def mutate(self, heat=1.0):
		return type(self)([i.mutate(heat) for i in self.v])


	def dist(self, other):
		zippy = zip(self.value, other.value)
		return np.linalg.norm([i - j for (i,j) in zippy])



def RandIntRangeParamOf(min, max):
	return lambda: RangeParam([
		RandIntParamOf(min, max)(),
		RandIntParamOf(min, max)(),
	])





class LRParam(GeneticParam):
	def __init__(self, v=None):
		sample = pow(10, random.uniform(-4, 2))
		self.v = v if v is not None else sample

	def mutate(self, heat=1.0):
		return type(self)(self.v * pow(10, heat*random.uniform(-0.5,0.5)))



class Heritage(GeneticParam):

	def vend(self):
		return random.choice(string.ascii_letters)

	def __init__(self, v=""):
		self.v = v + self.vend()

	def mutate(self, heat=1.0):
		return type(self)(self.v)

	def dist(self, other):
		return 0



""" Gives a fresh model folder name every mutation """
class ModelId(GeneticParam):

	def vend(self):
		return str(uuid.uuid4())

	def __init__(self, v=None):
		self.v = v if v is not None else {}

		# Ensure we have an id
		if "cur" not in self.v:
			self.v["cur"] = self.vend()

		if "warm_start_from" not in self.v:
			self.v["warm_start_from"] = None

	def mutate(self, heat):

		# cur_path = self.v.get("cur", None)
		# cur_exists = cur_path is not None and path_exists(cur_path)
		# warm_start_from = cur_path if cur_exists else self.v["warm_start_from"]
		# logger.debug("Mutate ModelId! {} {} {} {}".format(self.v, cur_path, cur_exists, warm_start_from))

		# I'd like to test that this directory does contain something, but it's
		# awkward to access args here. #injection
		warm_start_from = self.v["cur"]

		return type(self)({
			"cur": self.vend(),
			"warm_start_from": warm_start_from
		})

	def dist(self, other):
		return 0


class VariableParam(InitableParam):

	def __eq__(self, other):

		if self.v is None or other.v is None:
			return False

		for i in zip(self.v, other.v):
			if not np.array_equal(i[0], i[1]):
				return False

		return True

	def mutate(self, heat=1.0):
		return VariableParam(copy.copy(self.v))

	def __str__(self):
		return str(self.v.values())

	def dist(self, other):
		raise NotImplementedError()



class OptimizerParam(GeneticParam):
	def __init__(self, v=None):
		self.choices = [
				tf.train.AdamOptimizer,
				tf.train.RMSPropOptimizer,
				tf.train.GradientDescentOptimizer,
				tf.train.AdagradOptimizer
		]
		self.v = v if v is not None else random.choice(self.choices)

	def mutate(self, heat=1.0):
		o = self.value

		if random.random() > 1 - 0.2*heat:
			o = random.choice(self.choices)

		return type(self)(o)

	def dist(self, other):
		return 0 if self.value == other.value else 1
