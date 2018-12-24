
from typing import *

from abc import *
import numpy as np

RawTensor = Any


class FixedSizeTensor(NamedTuple):
	tensor:RawTensor
	size:List[int]

class Component(ABC):

	def __init__(self, args:Dict[str, Any], name:str=None):
		'''
		Components should instantiate their sub-components in init

		This is so that methods like print do not need to run
		`forward` (which may be impossible outside of a session)
		prior to being able to recursively print all sub-components.
		'''
		self.args = args
		self.name = name


	@abstractmethod
	def forward(self, features:Dict[str, RawTensor]) -> RawTensor:
		'''
		Wire the forward pass (e.g. take tensors and return transformed tensors)
		to ultimately build the whole network
		'''
		pass

	def taps(self) -> Dict[str, RawTensor]:
		'''
		Get the taps (tensors that provide insight into the workings
		of the network)

		Forward will always have been called before this method, so 
		it's ok to stash tensors as instance members in forward
		then recall them here
		'''
		return {}


	def tap_sizes(self) -> Dict[str, List[int]]:
		'''
		Get the names and sizes of expected taps

		Will be called independently from forward and taps
		'''
		return {}

	def print(self, tap_dict:Dict[str, np.array], path:List[str]):
		'''
		Print predict output nicely
		'''
		pass


	def _do_recursive_map(self, fn, path:List[str]=[]):
		new_path = [*path, self.name]
		new_path = [i for i in new_path if i is not None]
		
		r = fn(self, new_path)

		for i in vars(self):
			if isinstance(i,Component):
				r = {**r, **i._do_recursive_map(fn, new_path)}

		return r


	def all_taps(self) -> Dict[str,RawTensor]:

		def fn(self, path):		
			r = self.taps()
			r_prefixed = {'_'.join([*path, k]): v 
				for k,v in r.items()}
			return r_prefixed

		sizes = self.all_tap_sizes()
		taps = self._do_recursive_map(fn)

		sk = set(sizes.keys())
		tk = set(taps.keys())

		assert sk == tk, f"Set mismatch, in sizes but not taps: {sk - tk}, in taps but not sizes: {tk - sk}"

		return taps

	def all_tap_sizes(self) -> Dict[str, List[int]]:

		def fn(self, path):
			r = self.tap_sizes()
			r_prefixed = {'_'.join([*path, k]): v 
				for k,v in r.items()}

			return r_prefixed

		return self._do_recursive_map(fn)


	# You must call recursive_taps before this
	def print_all(self, tap_dict:Dict[str, np.array], path:List[str]=[]):
		
		def fn(self, path):
			t = self.tap_sizes()

			r = {
				k: tap_dict['_'.join([*path, k])]
				for k in t.keys()
			}

			self.print(r, path)
			return {}

		self._do_recursive_map(fn)




class Tensor(Component):
	def __init__(self, name=None):
		super().__init__(name)

	def bind(self, tensor:RawTensor):
		self.tensor = tensor

	def forward(self, features):
		return self.tensor

		