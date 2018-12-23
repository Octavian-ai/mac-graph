
from typing import *

from abc import ABC
import numpy as np

RawTensor = Any


class FixedSizeTensor(NamedTuple):
	tensor:RawTensor
	size:List[int]

class Component(ABC):

	def __init__(name:str):
		self.name = name

	@abstracmethod
	def forward(self, args:Dict[str, Any], features:Dict[str, RawTensor]):
		pass


	# Will always be called AFTER forward
	@abstracmethod
	def taps(self) -> Dict[str, FixedSizeTensor]:

	def recursive_taps(self, path:List[str]=[]) -> Dict[str,RawTensor]:
		new_path = [*path, name]
		prefix = '_'.join(new_path)
		
		r = taps()
		r_prefixed = {prefix+"_"+k:v for k,v in r.items()}

		self.taps_keys = [
			(prefix+"_"+k, k) for k in r.keys()
		]

		for i in var(self):
			if isinstance(i,Component):
				r = {**r, i.recursive_taps(new_path)}

		return r


	# Will always be called AFTER taps
	@abstracmethod
	def print(self, tap_dict:Dict[str, np.array], path:List[str]):
		"""Print predict output nicely"""
		pass


	def recursive_print(self, tap_dict:Dict[str, np.array], path:List[str]=[]):
		assert self.taps_keys is not None

		new_path = [*path, name]
		my_taps = { k : tap_dict[k_p] for (k_p, k) in self.taps_keys}

		self.print(my_taps, new_path)

		for i in var(self):
			if isinstance(i,Component):
				i.print(tap_dict, new_path)




class Tensor(Component):
	def __init__(tensor:RawTensor, name=None):
		self.tensor=tensor

	def forward(args, features):
		return self.tensor

	def taps(self):
		return {}

	def print(self, tap_dict, path):
		pass

		