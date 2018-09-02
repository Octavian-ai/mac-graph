
import random
import math

class Balancer(object):

	def __init__(self, partitioner, balance_freq, name=""):
		self.batch_i = 0
		self.partitioner = partitioner
		self.balance_freq = balance_freq
		self.name = name

	def oversample(self, n):
		'''Return over-sampling with n items total'''
		pass

	def add(self, doc, item):
		self.batch_i += 1
		self.pipe_if_ready()

	def pipe(self):
		for i in self.oversample(self.oversample_len()):
			self.partitioner.write(*i)

	def pipe_if_ready(self):
		if self.batch_i > self.balance_freq:
			self.pipe()
			self.batch_i = 0

	def __enter__(self):
		return self

	def __exit__(self, *vargs):
		self.pipe()




class ListBalancer(Balancer):

	def __init__(self, partitioner, balance_freq, name):
		super().__init__(partitioner, balance_freq, name)
		self.data = []

	def oversample_len(self):
		return len(self.data)

	def add(self, doc, item):
		self.data.append((doc,item))
		self.data = self.data[-self.balance_freq:]
		super().add(doc, item)

	def oversample(self, n):

		delta = n - len(self.data)

		print(f"ListBalancer sample len={len(self.data)}, n={n}, {self.name}")

		if len(self.data) == 0:
			raise ValueError("Cannot sample empty list")

		elif n >= len(self.data):
			# If bigger, insure every item is represented
			return self.data + [random.choice(self.data) for i in range(delta)]
		
		else:
			return [random.choice(self.data) for i in range(n)]



class DictBalancer(Balancer):

	def __init__(self, key_pred, CtrClzz, partitioner, balance_freq=50, name=""):
		super().__init__(partitioner, balance_freq, name)
		self.data = {}
		self.key_pred = key_pred
		self.CtrClzz = CtrClzz

	def oversample_len(self):
		def safe_max(n):
			return 0 if len(n) == 0 else max(n)

		return len(self.data) * safe_max([i.oversample_len() for i in self.data.values()])


	def add(self, doc, item):
		key = self.key_pred(doc)

		if key not in self.data:
			self.data[key] = self.CtrClzz(self.partitioner, self.balance_freq, key)

		self.data[key].add(doc, item)
		super().add(doc, item)

	def oversample(self, n):
		print(f"DictBalancer sample len={len(self.data)}, n={n}, {self.name}")

		n_per_class = math.ceil(n / len(self.data))
		r = []
		for k, v in self.data.items():
			o = v.oversample(n_per_class)
			assert len(o) >= n_per_class, f"Oversample({n_per_class}) only returned {len(o)} items. class={v.name}, {self.name}"
			r.extend(o)
		return r


class TwoLevelBalancer(DictBalancer):
	def __init__(self, key1, key2, partitioner, balance_freq):
		Inner = lambda partitioner, balance_freq, name: DictBalancer(key2, ListBalancer, partitioner, balance_freq, name)
		super().__init__(key1, Inner, partitioner, balance_freq, "TwoLevelBalancer")





# class Balancer(object):
# 	"""Streaming oversampler. Will only oversample classes seen in batch, bigger frequency is better"""

# 	def __init__(self, partitioner, hold_back=100):
# 		self.partitioner = partitioner
# 		self.total_classes = Counter()
# 		self.records_by_class = {}
# 		self.batch_i = 0
# 		self.hold_back = hold_back

# 	def __enter__(self, *vargs):
# 		return self

# 	def __exit__(self, *vargs):
# 		logger.debug(f"Classes after oversampling: {self.total_classes}")
# 		self.oversample()

# 	# --------------------------------------------------------------------------
# 	# Class balancing
# 	# --------------------------------------------------------------------------

# 	# Need to balance hierarchically by answer, then question type
# 	# Need a primative that is recursive and can
# 	# - Construct taking a storage class
# 	# - Store
# 	# - Sample
# 	# Make a base list wrapper, and a balancer

# 	def record_batch_item(self, doc, record):
# 		self.total_classes[doc["answer"]] += 1

# 		if doc["answer"] not in self.records_by_class:
# 			self.records_by_class[doc["answer"]] = []

# 		self.records_by_class[doc["answer"]].append(record)
# 		if len(self.records_by_class[doc["answer"]]) > self.hold_back:
# 			self.records_by_class[doc["answer"]] = self.records_by_class[doc["answer"]][-self.hold_back:]
		
# 		assert len(self.records_by_class[doc["answer"]]) <= self.hold_back

# 		self.batch_i += 1

# 	def oversample(self):
# 		if len(self.total_classes) > 0:
# 			target = max(self.total_classes.values())

# 			for key, count in self.total_classes.items():
# 				if count < target:
# 					delta = target - count
# 					logger.debug(f"Oversampling {key} x {delta}")
# 					for i in range(delta):
# 						self.partitioner.write(random.choice(self.records_by_class[key]))
# 						self.total_classes[key] += 1
						
# 			self.batch_classes = Counter()
# 			self.batch_i = 0

# 	def oversample_every(self, freq):
# 		if self.batch_i >= freq:
# 			self.oversample()




