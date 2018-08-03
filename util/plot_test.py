
import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import *

from .env import *


class PlotTestCase(unittest.TestCase):

	def run_ploty_test(self, ploty):
		try:
			os.remove(ploty.csv_file_path)
			os.remove(ploty.png_file_path)
		except FileNotFoundError:
			pass

		for i in range(10):
			ploty.add_result(i, i, 'id')

		ploty.write()

		self.assertTrue(os.path.isfile(ploty.csv_file_path))

		if ploty.is_png_enabled:
			self.assertTrue(os.path.isfile(ploty.png_file_path))


	def test_basics(self):
		test_args = gen_args()
		ploty = Ploty(test_args, 'test_basics')
		self.run_ploty_test(ploty)

	def test_gcs(self):
		test_args = gen_args('octavian-test', 'unittest')
		ploty = Ploty(test_args, 'test_gcs')
		self.run_ploty_test(ploty)




if __name__ == '__main__':
	tf.logging.set_verbosity('INFO')
	unittest.main()


