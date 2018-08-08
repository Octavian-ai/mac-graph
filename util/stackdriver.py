import logging
import sys
from pythonjsonlogger import jsonlogger


class StackdriverJsonFormatter(jsonlogger.JsonFormatter, object):
	def __init__(self, fmt="%(levelname) %(name) %(message)", style='%', *args, **kwargs):
		jsonlogger.JsonFormatter.__init__(self, fmt=fmt, *args, **kwargs)

	def process_log_record(self, log_record):
		log_record['severity'] = log_record['levelname']
		log_record['labels'] = {"module":log_record['name']}
		del log_record['levelname']
		del log_record['name']
		return super(StackdriverJsonFormatter, self).process_log_record(log_record)



def install(logger, level='DEBUG'):
	handler = logging.StreamHandler(sys.stdout)
	formatter = StackdriverJsonFormatter()
	handler.setFormatter(formatter)

	logger.addHandler(handler)
	logger.setLevel(level)