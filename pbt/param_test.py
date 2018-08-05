
import unittest
import os.path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .param import *


class ParamTestCase(unittest.TestCase):
	
	def test_int_param(self):
		p = IntParam(30,1,100)

		for i in range(20):
			self.assertIsInstance(p.value, int)
			p = p.mutate()


	def test_int_range_param(self):
		p = RandIntRangeParamOf(1, 5)()

		for i in range(40):
			self.assertGreaterEqual(p.value[0], 1)
			self.assertLessEqual(p.value[1], 5)
			# print(p.v[0].v, p.v[1].v, p.value)
			p = p.mutate()




if __name__ == '__main__':
	unittest.main(module='pbt',defaultTest="ParamTestCase")


