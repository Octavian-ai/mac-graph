
import logging, coloredlogs
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG',logger=logger)
coloredlogs.install(level='DEBUG',logger=logging.getLogger('pbt'))
coloredlogs.install(level='DEBUG',logger=logging.getLogger('experiment'))


import requests
import json
import platform
import time
import pika
import threading
import traceback

from .helpers import *
from .args import get_args

def i_am_supervisor(args):

	try:
		res = requests.get("http://localhost:4040")
	except requests.exceptions.ConnectionError:
		logger.warning("Could not contact leadership election sidecar, assuming not leader")
		return False

	if(res.ok):
		data = json.loads(res.content)
		leader_name = data["name"]
		my_name = platform.node()
		return leader_name == my_name

	else:
		res.raise_for_status()


def i_am_drone(args):
	am_sup = i_am_supervisor(args)
	am_drone = not am_sup or args.master_works
	return am_drone

# --------------------------------------------------------------------------
# Thread work loops
# --------------------------------------------------------------------------


def do_drone(args):
	drone = None

	try:
		while True:
			am_drone = i_am_drone(args)

			if am_drone and drone is None:
				logger.info("Start drone")
				drone = get_drone(args)	

			elif not am_drone and drone is not None:
				logger.info("Stop drone")
				drone.close()
				drone = None

			if drone is not None:
				drone.run_epoch()

			time.sleep(args.sleep_per_cycle)

	# TODO: actual signalling
	except KeyboardInterrupt:
		if drone is not None:
			drone.close()

	except Exception as e:
		traceback.print_exc()
		raise e

def do_supervisor(args):
	sup = None

	try:
		while True:
			am_sup = i_am_supervisor(args)

			if am_sup and sup is None:
				logger.info("Start supervisor")
				sup = get_supervisor(args)
			elif not am_sup and sup is not None:
				logger.info("Stop supervisor")
				sup.close()
				sup = None

			if sup is not None:
				sup.run_epoch()

			time.sleep(args.sleep_per_cycle)

	# TODO: actual signalling
	except KeyboardInterrupt:
		if sup is not None:
			sup.close()

	except Exception as e:
		traceback.print_exc()
		raise e


# --------------------------------------------------------------------------
# Dispatch threads from main loop
# --------------------------------------------------------------------------

def run_main_dispatch(args):
	my_sup = None
	my_drones = { k:None for k in range(args.n_drones) }

	try:
		while True:
			if my_sup is None or not my_sup.isAlive():
				logger.debug("Dispatch supervisor thread")
				my_sup = threading.Thread(target=do_supervisor, args=(args,))
				my_sup.setDaemon(True)
				my_sup.start()

			for key, drone in my_drones.items():
				if drone is None or not drone.isAlive():
					logger.debug("Dispatch drone thread")
					t = threading.Thread(target=do_drone, args=(args,))
					t.setDaemon(True)
					t.start()
					my_drones[key] = t

			time.sleep(args.sleep_per_cycle)

	except KeyboardInterrupt:
		# do something to signal to threads
		pass



if __name__ == "__main__":
	args = get_args()
	run_main_dispatch(args)
	

