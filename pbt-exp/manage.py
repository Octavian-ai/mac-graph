
import logging
logging.basicConfig()

from .helpers import *

if __name__ == '__main__':
	args = get_args()

	supervisor = get_supervisor(args)

	if args.load:
		supervisor.load()

	supervisor.manage()
